import random
import numpy as np
import math

import torch
import torch.nn as nn

import pyredner

class MeshRenderer(object):
    def __init__(self, mesh_path, out_sz, num_samples, device):
        super(MeshRenderer, self).__init__()
        self.device = device

        self.out_sz = out_sz
        self.camera = pyredner.Camera(
            position     = torch.tensor([0.0, 4.0, -4.0]),
            look_at      = torch.tensor([0.0, 0.0, 0.0]),
            up           = torch.tensor([0.0, 1.0, 0.0]),
            fov          = torch.tensor([45.0]),
            clip_near    = 0.01,
            resolution   = (self.out_sz, self.out_sz))

        self.materials = [
            pyredner.Material(diffuse_reflectance = torch.tensor(
                [1.0, 1.0, 1.0], device=self.device)),
            pyredner.Material(diffuse_reflectance = torch.tensor(
                [0.0, 0.0, 0.0], device=self.device)),
            ]

        self.shapes = [
            # TODO: GPU device choice is super sketchy
            pyredner.Shape(
                *pyredner.load_obj(mesh_path),
                0),
            pyredner.Shape(
                torch.tensor([
                    # on top
                    [-4.0,   7.0, -4.0],
                    [ 4.0,   7.0, -4.0],
                    [ 4.0,   7.0,  4.0],
                    [-4.0,   7.0,  4.0],
                    # on bottom
                    [-4.0,  -7.0, -4.0],
                    [ 4.0,  -7.0, -4.0],
                    [ 4.0,  -7.0,  4.0],
                    [-4.0,  -7.0,  4.0]],
                    device=self.device),
                torch.tensor([
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 5, 6],
                    [4, 6, 7]],
                    dtype = torch.int32,
                    device=self.device),
                None,
                None,
                1)]

        self.lights = [pyredner.Light(1, torch.tensor([2.0, 2.0, 2.0]))]

        self.num_samples = num_samples

    def __call__(self, sample):
        # Set Material
        # assumer sample is of type Tensor
        new_material = pyredner.Material(
                diffuse_reflectance = sample,
                specular_reflectance = torch.tensor(
                    [0.7, 0.7, 0.7], device=self.device),
                )
        self.materials[0] = new_material
        # Sample Camera Position
        d = 3.0 + random.uniform(0.0, 1.0) * 3.0
        phi = random.uniform(0.0, 1.0) * math.pi * 2
        theta = math.acos(1 - random.uniform(0.0, 1.0))
        px, py, pz = math.cos(phi)*math.sin(theta),\
                     math.cos(theta),\
                     math.sin(phi)*math.sin(theta)
        # Sample Camera Look-at
        lx, ly, lz = random.uniform(-1.0, 1.0),\
                     random.uniform(-1.0, 1.0),\
                     random.uniform(-1.0, 1.0)
        # Sample Camera Up-vector
        ux, uy, uz = random.uniform(-0.2, 0.2),\
                     1.0,\
                     random.uniform(-0.2, 0.2)

        self.camera = pyredner.Camera(
            position     = torch.tensor([px, py, pz]) * d,
            look_at      = torch.tensor([lx, ly, lz]),
            up           = torch.tensor([ux, uy, uz]),
            fov          = torch.tensor([45.0]),
            clip_near    = 0.01,
            resolution   = (self.out_sz, self.out_sz))

        # Define Scene
        scene = pyredner.Scene(
            self.camera, self.shapes, self.materials, self.lights)
        args = pyredner.RenderFunction.serialize_scene(
            scene = scene,
            num_samples = self.num_samples,
            max_bounces = 1)
        seed = random.randint(0, 255)
        return pyredner.RenderFunction.apply(seed, *args)

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
        new_material = pyredner.Material(
                diffuse_reflectance = sample,
                specular_reflectance = torch.tensor(
                    [0.8, 0.8, 1.0], device=self.device)
                )
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

class CornerCrop(object):
    def __init__(self, size):
        super(CornerCrop, self).__init__()
        self.size = size

    def __call__(self, input):
        return input[:self.size,:self.size,:]

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

class MeshRenderLayer(nn.Module):
    def __init__(self, mesh_path, out_sz, num_samples, device):
        super(MeshRenderLayer, self).__init__()
        self.renderer = MeshRenderer(mesh_path, out_sz, num_samples, device)

    def forward(self, input):
        out = self.renderer(input)
        return out

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

class CompositLayer(nn.Module):
    def __init__(self, bkgd, size, device):
        super(CompositLayer, self).__init__()
        #self.crop = RandomCrop(size)
        self.size = size
        self.device = device
        self.crop = CornerCrop(size)
        # TODO: this is probably not the best way to do this
        self.rng = bkgd is None

        if self.rng:
            self.bkgd = None
        else:
            self.bkgd = torch.tensor(self.crop(bkgd), device=device)

    def forward(self, input):
        # input is in format HWC
        if self.rng:
            bkgd_color = (random.uniform(0.0, 1.0),
                          random.uniform(0.0, 1.0),
                          random.uniform(0.0, 1.0))
            self.bkgd = torch.tensor(bkgd_color, device=self.device)\
                             .expand(self.size, self.size, len(bkgd_color))
        alpha = input[:,:,-1:]
        return alpha * input[:,:,:-1] + (1 - alpha) * self.bkgd

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

class NormalizedCompositLayer(nn.Module):
    def __init__(self, bkgd, out_sz, device):
        super(NormalizedCompositLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                CompositLayer(bkgd, out_sz, device),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out

class NormalizedRenderLayer(nn.Module):
    def __init__(self, mesh_path, bkgd, out_sz, num_samples, device):
        super(NormalizedRenderLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                CompositLayer(bkgd, out_sz, device),
                MeshRenderLayer(mesh_path, out_sz, num_samples, device),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out
