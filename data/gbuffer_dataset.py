import random
import re
import math
import os.path

import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.image_util import imread, imwrite
from util.transform_util import GaussianNoiseNP, ToTensor


class GbufferDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--gbuffer_root', type=str, default='./datasets/gbuffers/', help='Root directory for gbuffers')
        parser.add_argument('--gaussian_sigma', type=float, default=0.0, help='STD for Gaussian noise to applied on dataset')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # Load target images and their alphamasks
        self.dir_target_img = os.path.join(opt.dataroot, 'img')
        self.target_img_paths = make_dataset(self.dir_target_img)
        self.target_img_paths = sorted(self.target_img_paths)
        self.target_img_size = len(self.target_img_paths)

        self.dir_target_mask = os.path.join(opt.dataroot, 'mask')
        self.target_mask_paths = make_dataset(self.dir_target_mask)
        self.target_mask_paths = sorted(self.target_mask_paths)
        self.target_mask_size = len(self.target_mask_paths)

        assert(self.target_img_size == self.target_mask_size)

        self.dir_gbuffer_position = os.path.join(opt.gbuffer_root, 'position')
        self.dir_gbuffer_mask = os.path.join(opt.gbuffer_root, 'mask')

        self.noise = GaussianNoiseNP(opt.gaussian_sigma)

        self.image_transform = get_transform(opt)
        self.mask_transform = ToTensor()

    def __getitem__(self, index):
        # Load Target Image and alphamask
        target_img_path = self.target_img_paths[index % self.target_img_size]
        target_mask_path = self.target_mask_paths[index % self.target_mask_size]

        target_img = imread(target_img_path)
        target_mask = imread(target_mask_path)

        target_img = self.noise(target_img)

        target = target_img * target_mask
        target = self.image_transform(target)

        target_mask = np.repeat(target_mask[:,:,:1], 4, axis=-1)
        target_mask = self.mask_transform(target_mask)

        # Load Gbuffer Images and alphamask
        # TODO: For now hardcoded for one learneable texture
        gbuffer_position_path = os.path.join(self.dir_gbuffer_position, 'octavia_clean.png')
        gbuffer_mask_path = os.path.join(self.dir_gbuffer_mask, 'octavia_clean.png')

        # Setup gbuffer
        gbuffer_position = imread(gbuffer_position_path)
        gbuffer_mask= imread(gbuffer_mask_path)

        gbuffer = gbuffer_position * gbuffer_mask
        gbuffer = self.image_transform(gbuffer)
        #gbuffer = torch.cat([gbuffer_position, gbuffer_normal], dim=0)

        gbuffer_mask = np.repeat(gbuffer_mask[:,:,:1], 4, axis=-1)
        gbuffer_mask = self.mask_transform(gbuffer_mask)

        # Fetch image path for config loading during rendering
        config_key = re.split('/|\.', target_img_path)[-2]

        if self.opt.input_nc == 1:  # RGB to gray
            tmp = gbuffer[0, ...] * 0.299 + gbuffer[1, ...] * 0.587 + gbuffer[2, ...] * 0.114
            gbuffer = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:  # RGB to gray
            tmp = target[0, ...] * 0.299 + target[1, ...] * 0.587 + target[2, ...] * 0.114
            target = tmp.unsqueeze(0)
        return {'gbuffer': gbuffer, 'gbuffer_mask': gbuffer_mask,
                'target': target, 'target_mask': target_mask,
                'target_paths': target_img_path, 'config_keys': config_key}

    def __len__(self):
        return self.target_img_size

    def name(self):
        return 'GbufferDataset'
