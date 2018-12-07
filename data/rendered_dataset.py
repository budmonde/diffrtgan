import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random


class RenderedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        #TODO setup A

        self.dir_B = os.path.join(opt.dataroot)
        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        inp_dims = (self.opt.fineSize, self.opt.fineSize, self.opt.input_nc)
        # TODO allow for seed choice
        noise = np.random.uniform(0.0, 1.0, inp_dims)
        A_img = Image.fromarray(np.uint8(noise * 255))

        B_path = self.B_paths[index % self.B_size]
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        # TODO questionable if this should feature should be kept
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'B_paths': B_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'RenderedDataset'
