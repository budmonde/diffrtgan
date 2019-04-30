import itertools
import json
import os

import numpy as np
import torch
import torch.nn as nn

from . import networks
from .base_model import BaseModel
from util.image_pool import ImagePool
from util.image_util import imread, grey2heatmap
from util.render_util import RenderConfig
from util.sample_util import *
from util.torch_util import *


class StableModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
                pool_size=0,
                no_flip=True,
                no_dropout=True,
                display_ncols=5,
                input_nc=6,
                load_size=256,
        )
        # Visuals Config
        parser.add_argument('--viz_composit_bkgd_path', type=str, default='./datasets/textures/composit_background.png', help='Compositing background used for visualization of semi-transparent textures')

        # Network Config
        parser.add_argument('--texture_nc', type=int, default=4, help='Number of channels in the texture output')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_tex = ['gbuffer_position', 'gbuffer_normal', 'synth_tex_show']
        visual_names_render = ['target', 'synth']
        visual_names_loss = ['heatmap_real', 'heatmap_fake']
        self.visual_names = visual_names_tex + visual_names_render + visual_names_loss

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks

        ### Begin Generator Pipeline ###

        # 1. Generate Texture from Gbuffer prior
        self.netG = networks.define_G(opt.input_nc, opt.texture_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 2. Pre-processing:
        #    - Strip batch dimension
        #    - Normalize to [0,1] domain
        #    - Switch dimension order
        #    - Mask out unlearneable texture values **applied at runtime**
        self.pre_process = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
        )

        # 3. Render Image
        self.scene_dict = json.loads(open(os.path.join(opt.dataroot, 'data.json')).read())
        self.render_config = RenderConfig()
        self.override = ConfigSampler({
            #'envmap_path'   : PathSamplerFactory(
            #    './datasets/envmaps/rasters', ext='exr'),
            'envmap_rotation'    : UniformSamplerFactory(0.0, 1.0),
        })

        self.render        = RenderLayer(self.render_config, self.device)

        # 4. Post-processing:
        #    - Add Signal Noise
        #    - Composit Alpha layer to mask out environment map
        #    - Undo switch dimension order
        #    - Undo normalization
        #    - Add back batch dimension
        noise_kwargs = {
            "sigma": opt.gaussian_sigma,
            "device": self.device,
        }
        composit_kwargs = {
            "background": np.array([[[0.0, 0.0, 0.0]]]),
            "size": opt.crop_size,
            "device": self.device,
        }
        self.post_process = nn.Sequential(
            GaussianNoiseLayer(**noise_kwargs),
            CompositLayer(**composit_kwargs),
            HWC2CHWLayer(),
            NormalizeLayer(0.5, 0.5),
            AddBatchDimLayer(),
        )

        background = imread(opt.viz_composit_bkgd_path)

        visdom_kwargs = {
            "background": background,
            "size": opt.crop_size,
            "device": self.device,
        }
        self.composit_layer = NormalizedCompositLayer(**visdom_kwargs)

        ### End Generator Pipeline ###

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.synth_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.gbuffer = input['gbuffer'].to(self.device)
        self.target = input['target'].to(self.device)
        self.gbuffer_mask = input['gbuffer_mask'].to(self.device)[0,...]
        self.target_mask = input['target_mask'].to(self.device)[0,...]
        self.disc_mask = input['disc_mask'].to(self.device)[0,...]
        self.config_key = input['config_keys'][0]

        # Process visuals
        self.gbuffer_position = self.gbuffer[:,0:3,...]
        self.gbuffer_normal = self.gbuffer[:,3:6,...]

    def forward(self):
        self.synth_tex = self.netG(self.gbuffer)
        # Pre-process
        self.synth_tex = self.pre_process(self.synth_tex)
        self.synth_tex = self.synth_tex * self.gbuffer_mask
        # Set camera parameters and pass to renderer
        scene = self.scene_dict[self.config_key]
        scene_override = self.override.generate()
        for k, v in scene_override.items():
            scene[k] = v
        self.render_config.set_scene(scene)
        self.synth = self.render(self.synth_tex)
        # Post-process
        self.synth = torch.cat([self.synth, self.target_mask], dim=-1)
        self.synth = self.post_process(self.synth)

        # Process visuals
        with torch.no_grad():
            self.synth_tex_show = self.synth_tex.clone()
            self.synth_tex_show[:,:,-1] = self.synth_tex_show[:,:,-1] + (1 - self.synth_tex_show[:,:,-1]) * (1 - self.gbuffer_mask[:,:,0])
            self.synth_tex_show = self.composit_layer(self.synth_tex_show)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        #pred_real = pred_real * self.disc_mask
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        #pred_fake = pred_fake * self.disc_mask
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()

        # Visualize heatmap
        with torch.no_grad():
            self.heatmap_real = grey2heatmap(pred_real.clone(), self.opt.crop_size)
            self.heatmap_fake = grey2heatmap(pred_fake.clone(), self.opt.crop_size)

        return loss_D

    def backward_D(self):
        synth = self.synth_pool.query(self.synth)
        self.loss_D = self.backward_D_basic(
                self.netD, self.target, synth)

    def backward_G(self):
        # GAN loss D(G(A))
        self.loss_G = self.criterionGAN(self.netD(self.synth), True)
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
