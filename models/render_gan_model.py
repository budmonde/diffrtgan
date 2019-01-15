import torch
import numpy as np
import itertools
from util.image_pool import ImagePool
from util.image_util import imread
from util.render_util import NormalizedRenderLayer, NormalizedCompositLayer
from .base_model import BaseModel
from . import networks


class RenderGANModel(BaseModel):
    def name(self):
        return 'RenderGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(input_nc=4)
        parser.add_argument('--mesh_path', type=str, default='./datasets/meshes/octavia/octavia_uv.obj', help='Path of mesh to render')
        # TODO: fix image transform so that it looks at texture_nc for dim#
        parser.add_argument('--texture_nc', type=int, default=4, help='Number of channels in the texture output')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_tex = ['real_tex_A_show', 'real_tex_B_show', 'fake_tex_B_show']
        visual_names_render = ['real_render_B', 'fake_render_B']
        self.visual_names = visual_names_tex + visual_names_render

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.texture_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        bkgd_color = (0.7, 0.0, 0.7)
        #bkgd = torch.tensor(bkgd_color)\
        #            .expand(opt.fineSize, opt.fineSize, len(bkgd_color))
        bkgd = None

        vis_bkgd = torch.tensor(imread('./datasets/textures/transparency/transparency.png'), dtype=torch.float32)

        # TODO make num samples configurable
        self.render_layer = NormalizedRenderLayer(opt.mesh_path, bkgd, opt.fineSize, 4, self.device)
        self.composit_layer = NormalizedCompositLayer(vis_bkgd, opt.fineSize, self.device)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_render_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_tex_A = input['A'].to(self.device)
        self.real_tex_B = input['B'].to(self.device)
        self.real_render_B = self.render_layer(self.real_tex_B)

        # Composit for visuals
        self.real_tex_A_show = self.composit_layer(self.real_tex_A)
        self.real_tex_B_show = self.composit_layer(self.real_tex_B)

    def forward(self):
        self.fake_tex_B = self.netG(self.real_tex_A)
        self.fake_render_B = self.render_layer(self.fake_tex_B)

        # Composit for visuals
        self.fake_tex_B_show = self.composit_layer(self.fake_tex_B)

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
        fake_render_B = self.fake_render_B_pool.query(self.fake_render_B)
        self.loss_D = self.backward_D_basic(
                self.netD, self.real_render_B, fake_render_B)

    def backward_G(self):
        # GAN loss D(G(A))
        self.loss_G = self.criterionGAN(self.netD(self.fake_render_B), True)
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
