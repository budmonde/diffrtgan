import random
import numpy as np

import torch
import torch.nn as nn

import pyredner

class PlaneRenderer(object):
    def __init__(self, out_sz, num_samples, device):
        super(PlaneRenderer, self).__init__()
        self.device = device

        self.camera = pyredner.Camera(
            position     = torch.tensor([0.0, 0.0, -5.0]),
            look_at      = torch.tensor([0.0, 0.0, 0.0]),
            up           = torch.tensor([0.0, 1.0, 0.0]),
            fov          = torch.tensor([45.0]),
            clip_near    = 0.01,
            resolution   = (out_sz, out_sz))

        self.materials = [
            pyredner.Material(diffuse_reflectance = torch.tensor(
                [1.0, 1.0, 1.0], device=self.device)),
            pyredner.Material(diffuse_reflectance = torch.tensor(
                [0.0, 0.0, 0.0], device=self.device)),
            ]

        self.shapes = [
            pyredner.Shape(
                torch.tensor([
                    [-2.1, -2.1, 8.0],
                    [-2.1, 2.1, 0.0],
                    [2.1, -2.1, 8.0],
                    [2.1, 2.1, 0.0]],
                    device=self.device),
                torch.tensor([
                    [0, 1, 2],
                    [1, 3, 2]],
                    dtype = torch.int32,
                    device=self.device),
                torch.tensor([
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0]],
                    device=self.device),
                None,
                0),
            pyredner.Shape(
                torch.tensor([
                    [-1.0, -1.0, -7.0],
                    [1.0, -1.0, -7.0],
                    [-1.0, 1.0, -7.0],
                    [1.0, 1.0, -7.0]],
                    device=self.device),
                torch.tensor([
                    [0, 1, 2],
                    [1, 3, 2]],
                    dtype = torch.int32,
                    device=self.device),
                None,
                None,
                1)]

        self.lights = [pyredner.Light(1, torch.tensor([40.0, 40.0, 40.0]))]

        self.num_samples = num_samples

    def __call__(self, sample):
        # assumer sample is of type Tensor
        new_material = pyredner.Material(diffuse_reflectance = sample)
        self.materials[0] = new_material
        scene = pyredner.Scene(
            self.camera, self.shapes, self.materials, self.lights)
        args = pyredner.RenderFunction.serialize_scene(
            scene = scene,
            num_samples = self.num_samples,
            max_bounces = 1)
        seed = random.randint(0, 255)
        return pyredner.RenderFunction.apply(seed, *args)

class RandomCrop(object):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def __call__(self, input):
        y = np.random.choice(input.shape[0] - self.size)
        x = np.random.choice(input.shape[1] - self.size)
        return input[y : y + self.size, x : x + self.size, : ]

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

class PlaneRenderLayer(nn.Module):
    def __init__(self, out_sz, num_samples, device):
        super(PlaneRenderLayer, self).__init__()
        self.renderer = PlaneRenderer(out_sz, num_samples, device)

    def forward(self, input):
        out = self.renderer(input)
        return out

class HWC2CHWLayer(nn.Module):
    def __init__(self):
        super(HWC2CHWLayer, self).__init__()
        pass

    def forward(self, input):
        return input.transpose(1, 2).transpose(0, 1).contiguous()

class CHW2HWCLayer(nn.Module):
    def __init__(self):
        super(CHW2HWCLayer, self).__init__()
        pass

    def forward(self, input):
        return input.transpose(0, 1).transpose(1, 2).contiguous()

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        return (input - self.mean) / self.std

class StripBatchDimLayer(nn.Module):
    def __init__(self):
        super(StripBatchDimLayer, self).__init__()
        pass

    def forward(self, input):
        return input[0]

class AddBatchDimLayer(nn.Module):
    def __init__(self):
        super(AddBatchDimLayer, self).__init__()
        pass

    def forward(self, input):
        return input.unsqueeze(0)

class NormalizedRenderLayer(nn.Module):
    def __init__(self, out_sz, num_samples, device):
        super(NormalizedRenderLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                PlaneRenderLayer(out_sz, num_samples, device),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out
