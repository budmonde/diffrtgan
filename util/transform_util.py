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
        return lambd(input)

class Resize(object):
    def __init__(self, size):
        super(Resize, self).__init__()
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, input):
        return resize(input, self.size)

class RandomCrop(object):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def __call__(self, input):
        y = np.random.choice(input.shape[0] - self.size)
        x = np.random.choice(input.shape[1] - self.size)
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
