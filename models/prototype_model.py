import itertools
import json
import os

import numpy as np
import torch

from . import networks
from .base_model import BaseModel
from util.image_pool import ImagePool
from util.image_util import imread
from util.render_util import RenderConfig
from util.torch_util import NormalizedRenderLayer, NormalizedCompositLayer


class PrototypeModel(BaseModel):
    def name(self):
        return 'PrototypeModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
                pool_size=0,
                no_flip=True,
                no_dropout=True,
                display_ncols=6,
                input_nc=6,
                loadSize=256,
        )
        # Render Config
        parser.add_argument('--meshes_path', type=str, default='./datasets/meshes/one_mtl', help='Path of mesh pool to render')
        parser.add_argument('--envmaps_path', type=str, default='./datasets/envmaps/rasters', help='Path of envmap pool to render')
        parser.add_argument('--mc_subsampling', type=int, default=4, help='Number of Monte-Carlo subsamples per-pixel on rendering step')
        parser.add_argument('--mc_max_bounces', type=int, default=2, help='Max number of Monte-Carlo bounces ray on rendering step')

        # Visuals Config
        parser.add_argument('--viz_composit_bkgd_path', type=str, default='./datasets/textures/transparency/transparency.png', help='Compositing background used for visualization of semi-transparent textures')

        # Network Config
        parser.add_argument('--texture_nc', type=int, default=4, help='Number of channels in the texture output')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0,
                            help='weight for cycle loss (B -> A -> B)')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A_position', 'real_A_normal', 'fake_B_tex_show', 'fake_B', 'rec_A_position', 'rec_A_normal']
        visual_names_B = ['real_B_copy', 'real_B', 'fake_A_position', 'fake_A_normal', 'rec_B_tex_show', 'rec_B']
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # G_A: A->B
        self.netG_A = networks.define_G(opt.input_nc, opt.texture_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # G_B: B->A
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        self.render_config = RenderConfig(json.loads(open(os.path.join(opt.dataroot, 'data.json')).read()))

        render_kwargs = {
            "meshes_path": opt.meshes_path,
            "envmaps_path": opt.envmaps_path,
            "out_sz": opt.fineSize,
            "num_samples": opt.mc_subsampling,
            "max_bounces": opt.mc_max_bounces,
            "device": self.device,
            "logger": None,
            "config": self.render_config,
        }
        composit_kwargs = {
            "background": np.array([[[0.0, 0.0, 0.0]]]),
            "size": opt.fineSize,
            "device": self.device,
        }
        self.render_fake = NormalizedRenderLayer(render_kwargs, composit_kwargs)
        self.render_rec = NormalizedRenderLayer(render_kwargs, composit_kwargs)

        background = imread(opt.viz_composit_bkgd_path)

        visdom_kwargs = {
            "background": background,
            "size": opt.fineSize,
            "device": self.device,
        }
        self.composit_layer = NormalizedCompositLayer(**visdom_kwargs)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_copy = self.real_B
        self.filename = input['filename']

        # For a given pair of A and B, there is one optimized configuration
        # that predicts the camera and scene geometry
        self.render_config.set_config(self.filename[0])

    def forward(self):
        # real_A -> fake_B_tex
        self.fake_B_tex = self.netG_A(self.real_A)
        # fake_B_tex -> fake_B
        self.fake_B = self.render_fake(self.fake_B_tex)
        # fake_B -> rec_A
        self.rec_A = self.netG_B(self.fake_B)

        # real_B -> fake_A
        self.fake_A = self.netG_B(self.real_B)
        # fake_A -> rec_B_tex
        self.rec_B_tex = self.netG_A(self.fake_A)
        # rec_B_tex -> rec_B
        self.rec_B = self.render_rec(self.rec_B_tex)

        # Separate gbuffer visuals
        self.real_A_position = self.real_A[:,:3,:,:]
        self.real_A_normal = self.real_A[:,3:,:,:]
        self.rec_A_position = self.rec_A[:,:3,:,:]
        self.rec_A_normal = self.rec_A[:,3:,:,:]
        self.fake_A_position = self.fake_A[:,:3,:,:]
        self.fake_A_normal = self.fake_A[:,3:,:,:]

        # Composit texture visuals
        self.fake_B_tex_show = self.composit_layer(self.fake_B_tex)
        self.rec_B_tex_show = self.composit_layer(self.rec_B_tex)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()