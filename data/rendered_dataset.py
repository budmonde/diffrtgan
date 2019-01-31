import random
import os.path

import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.image_util import imread


class RenderedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_B = os.path.join(opt.dataroot)
        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # CHW order
        inp_dims = (self.opt.input_nc, self.opt.fineSize, self.opt.fineSize)
        A = torch.tensor(np.random.uniform(-0.5, 0.5, inp_dims), dtype=torch.float32)

        B_path = self.B_paths[index % self.B_size]
        B_img = imread(B_path)
        B = self.transform(B_img)

        if self.opt.input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'B_paths': B_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'RenderedDataset'
