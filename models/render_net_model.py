import itertools
import json

import numpy as np
import torch

from . import networks
from .base_model import BaseModel
from util.image_pool import ImagePool
from util.image_util import imread
from util.render_util import RenderConfig
from util.torch_util import NormalizedRenderLayer, NormalizedCompositLayer


class RenderNetModel(BaseModel):
    def name(self):
        return 'RenderNetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(input_nc=4, no_flip=True, loadSize=256)
        parser.add_argument('--meshes_path', type=str, default='./datasets/meshes/one_mtl', help='Path of mesh pool to render')
        parser.add_argument('--envmaps_path', type=str, default='./datasets/envmaps/rasters', help='Path of envmap pool to render')
        parser.add_argument('--texture_nc', type=int, default=4, help='Number of channels in the texture output')
        parser.add_argument('--mc_subsampling', type=int, default=4, help='Number of Monte-Carlo subsamples per-pixel on rendering step')
        parser.add_argument('--mc_max_bounces', type=int, default=2, help='Max number of Monte-Carlo bounces ray on rendering step')
        parser.add_argument('--config_path', type=str, default='./datasets/empty.json', help='Path to camera configs for target images')

        parser.add_argument('--viz_composit_bkgd_path', type=str, default='./datasets/textures/transparency/transparency.png', help='Compositing background used for visualization of semi-transparent textures')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_tex = ['real_A_tex_show', 'fake_B_tex_show']
        visual_names_render = ['real_B', 'fake_B_render']
        self.visual_names = visual_names_tex + visual_names_render

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.texture_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        render_background = None
        visdom_background = torch.tensor(imread(opt.viz_composit_bkgd_path), dtype=torch.float32)

        self.render_config = RenderConfig(json.loads(open(opt.config_path).read()))

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
            "background": render_background,
            "size": opt.fineSize,
            "device": self.device,
        }
        self.render_layer = NormalizedRenderLayer(render_kwargs, composit_kwargs)

        visdom_composit_kwargs = {
            "background": render_background,
            "size": opt.fineSize,
            "device": self.device,
        }
        self.composit_layer = NormalizedCompositLayer(**visdom_composit_kwargs)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_B_render_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A_tex = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_name = input['B_name']

        # Composit for visuals
        self.real_A_tex_show = self.composit_layer(self.real_A_tex)

    def forward(self):
        self.fake_B_tex = self.netG(self.real_A_tex)
        # Set camera parameters and pass to renderer
        self.fake_B_id = self.render_config.set_config(self.real_B_name[0])
        self.fake_B_render = self.render_layer(self.fake_B_tex)

        # Composit for visuals
        self.fake_B_tex_show = self.composit_layer(self.fake_B_tex)

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

    def backward_D(self):
        fake_B_render = self.fake_B_render_pool.query(self.fake_B_render)
        self.loss_D = self.backward_D_basic(
                self.netD, self.real_B, fake_B_render)

    def backward_G(self):
        # GAN loss D(G(A))
        self.loss_G = self.criterionGAN(self.netD(self.fake_B_render), True)
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
