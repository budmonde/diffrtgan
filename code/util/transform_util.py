import random

import numpy as np
from PIL import Image
from skimage.transform import resize

import torch


class Debug(object):
    def __init__(self):
        super(Debug, self).__init__()
        pass

    def __call__(self, input):
        print(input.shape)
        return input

class GaussianNoise(object):
    # NOTE: Assumes input is HWC Formatted Image
    #       If input has 4 channels, it will ignore the alpha channel
    def __init__(self, sigma, device):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0, dtype=torch.float32, device=device)

    def __call__(self, input):
        if self.sigma != 0:
            scale = self.sigma * input.detach()
            # no noise in alpha channel
            if input.size()[-1] == 4:
                scale[:,:,-1] *= 0
            sampled_noise = self.noise.repeat(*input[...,:1].shape).normal_() * scale
            input = input + sampled_noise
        return input

class GaussianNoiseNP(object):
    # TODO: For some reason this one looks a little brighter than the torch version(?)
    def __init__(self, sigma):
        super(GaussianNoiseNP, self).__init__()
        self.sigma = sigma

    def __call__(self, input):
        scale = self.sigma * input
        sampled_noise = np.random.normal(np.zeros(input[...,:1].shape), 1.0) * scale
        out = input + sampled_noise
        return out

class NP2PIL(object):
    def __init__(self):
        super(NP2PIL, self).__init__()
        pass

    def __call__(self, input):
        return Image.fromarray(np.uint8(input * 255))

class PIL2NP(object):
    def __init__(self):
        super(PIL2NP, self).__init__()
        pass

    def __call__(self, input):
        return np.array(input) / 255.

class Composit(object):
    def __init__(self, background, size, device):
        super(Composit, self).__init__()
        self.device = device
        self.background = torch.tensor(
                Resize(size)(background),
                dtype=torch.float32,
                device=device)

    def __call__(self, input):
        # input is in format HWC
        alpha = input[:,:,-1:]
        return alpha * input[:,:,:-1] + (1 - alpha) * self.background

class Compose(object):
    def __init__(self, transform_list):
        super(Compose, self).__init__()
        self.transform_list = transform_list

    def __call__(self, input):
        output = input
        for tsfm in self.transform_list:
            output = tsfm(output)
        return output

class Lambda(object):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def __call__(self, input):
        return self.lambd(input)

class Resize(object):
    def __init__(self, size, order = 1):
        super(Resize, self).__init__()
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size
        self.order = order

    def __call__(self, input):
        return resize(input, self.size, order = self.order)

class RandomCrop(object):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def __call__(self, input):
        y = random.randint(0, input.shape[0] - self.size)
        x = random.randint(0, input.shape[1] - self.size)
        return input[y : y + self.size, x : x + self.size, : ]

class CornerCrop(object):
    def __init__(self, size):
        super(CornerCrop, self).__init__()
        self.size = size

    def __call__(self, input):
        return input[:self.size,:self.size,:]

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, input):
        if random.uniform(0.0, 1.0) < self.p:
            return np.flip(input, 1)
        else:
            return input

class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()
        pass

    def __call__(self, input):
        return torch.FloatTensor(np.array(input))

class HWC2CHW(object):
    def __init__(self):
        super(HWC2CHW, self).__init__()
        pass

    def __call__(self, input):
        return input.transpose(1, 2).transpose(0, 1).contiguous()

class CHW2HWC(object):
    def __init__(self):
        super(CHW2HWC, self).__init__()
        pass

    def __call__(self, input):
        return input.transpose(0, 1).transpose(1, 2).contiguous()

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input - self.mean) / self.std

class StripBatchDim(object):
    def __init__(self):
        super(StripBatchDim, self).__init__()
        pass

    def __call__(self, input):
        return input[0]

class AddBatchDim(object):
    def __init__(self):
        super(AddBatchDim, self).__init__()
        pass

    def __call__(self, input):
        return input.unsqueeze(0)
