import random
import re
import os.path

import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.image_util import imread


class MaskedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mask_applied', action='store_true', help='Check if mask has already been applied to initial image')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_B = os.path.join(opt.dataroot, 'img')
        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)
        self.B_size = len(self.B_paths)

        self.apply_mask = not opt.mask_applied
        if self.apply_mask:
            self.dir_B_mask = os.path.join(opt.dataroot, 'mask')
            self.B_mask_paths = make_dataset(self.dir_B_mask)
            self.B_mask_paths = sorted(self.B_mask_paths)
            self.B_mask_size = len(self.B_mask_paths)

            assert(self.B_mask_size == self.B_size)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # CHW order
        inp_dims = (self.opt.input_nc, self.opt.fineSize, self.opt.fineSize)
        A = torch.tensor(np.random.uniform(-0.5, 0.5, inp_dims), dtype=torch.float32)

        B_path = self.B_paths[index % self.B_size]
        B_name = re.split('/|\.', B_path)[-2]
        B_img = imread(B_path)

        if self.apply_mask:
            B_mask_path = self.B_mask_paths[index % self.B_mask_size]
            B_mask = imread(B_mask_path)
            B_img = B_img * B_mask

        B = self.transform(B_img)

        if self.opt.input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'B_paths': B_path, 'B_name': B_name}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'MaskedDataset'
