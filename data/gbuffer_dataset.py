import random
import re
import math
import os.path

import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.image_util import imread, imwrite
from util.transform_util import GaussianNoiseNP


class GbufferDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--gaussian_sigma', type=float, default=0.5, help='STD for Gaussian noise to applied on dataset')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_img = os.path.join(opt.dataroot, 'img')
        self.img_paths = make_dataset(self.dir_img)
        self.img_paths = sorted(self.img_paths)
        self.img_size = len(self.img_paths)

        self.dir_mask = os.path.join(opt.dataroot, 'mask')
        self.mask_paths = make_dataset(self.dir_mask)
        self.mask_paths = sorted(self.mask_paths)
        self.mask_size = len(self.mask_paths)

        self.dir_position = os.path.join(opt.dataroot, 'position')
        self.position_paths = make_dataset(self.dir_position)
        self.position_paths = sorted(self.position_paths)
        self.position_size = len(self.position_paths)

        self.dir_normal = os.path.join(opt.dataroot, 'normal')
        self.normal_paths = make_dataset(self.dir_normal)
        self.normal_paths = sorted(self.normal_paths)
        self.normal_size = len(self.normal_paths)

        assert(self.img_size == self.mask_size)
        assert(self.mask_size == self.position_size)
        assert(self.position_size == self.normal_size)

        self.noise = GaussianNoiseNP(opt.gaussian_sigma)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        B_img_path = self.img_paths[index % self.img_size]
        B_mask_path = self.mask_paths[index % self.mask_size]
        A_position_path = self.position_paths[index % self.position_size]
        A_normal_path = self.normal_paths[index % self.normal_size]

        # Setup A
        #A_position = imread(A_position_path)
        #A_normal = imread(A_normal_path)

        #A_position = self.transform(A_position)
        #A_normal = self.transform(A_normal)

        #A = torch.cat([A_position, A_normal], dim=0)
        A = generate_spherical_pos(self.opt.fineSize)
        A = self.transform(A)

        # Setup B
        B_img = imread(B_img_path)
        B_mask = imread(B_mask_path)

        B_img = self.noise(B_img)

        B = B_img * B_mask
        B = self.transform(B)

        # Setup B_paths, filename
        B_path = B_img_path
        filename = re.split('/|\.', B_img_path)[-2]

        if self.opt.input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'B_paths': B_path, 'filename': filename}

    def __len__(self):
        return self.img_size

    def name(self):
        return 'GbufferDataset'

def generate_spherical_pos(size):
    one = np.ones(size)[:,np.newaxis]
    samples = np.linspace(0, math.pi, size)[:,np.newaxis]
    cos = np.cos(samples)
    sin = np.sin(samples)
    out = np.zeros((size, size, 3))
    out[:,:,0] = cos @ np.transpose(one)
    out[:,:,1] = sin @ np.transpose(sin)
    out[:,:,2] = sin @ np.transpose(cos)
    return out
