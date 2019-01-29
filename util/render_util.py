import os
import random
import numpy as np
import math

import torch
import torch.nn as nn

import pyredner

from .transform_util import *

def get_children_path_list(path):
    return ['{}/{}'.format(path, f) for f in os.listdir(path)]

class MeshRenderer(object):
    def __init__(self, meshes_path, out_sz, num_samples, device):
        super(MeshRenderer, self).__init__()
        self.fov = 45.0
        self.clip_near = 0.01
        self.resolution = (out_sz, out_sz)
        self.num_samples = num_samples
        self.device = device

        def pos_rand():
            #d = 3.0 + random.uniform(1.0, 1.0) * 3.0
            #phi = random.uniform(0.25, 0.25) * math.pi * 2
            #theta = math.acos(1 - random.uniform(1.0, 1.0))
            #x, y, z = math.cos(phi)*math.sin(theta)*d,\
            #          math.cos(theta)*d,\
            #          math.sin(phi)*math.sin(theta)*d
            #return (x, y, z)
            return (0.0, 1.5, random.choice([-9.0, 9.0]))
        def look_rand():
            #x, y, z = random.uniform(-1.0, 1.0),\
            #          random.uniform(-1.0, 1.0,\
            #          random.uniform(-1.0, 1.0)
            #return (x, y, z)
            return (0.0, 1.5, 0.0)
        def up_rand():
            #x, y, z = random.uniform(-0.2, 0.2),\
            #          1.0,\
            #          random.uniform(-0.2, 0.2)
            #return (x, y, z)
            return (0.0, 1.0, 0.0)
        def rad_rand():
            return (2.0, 2.0, 2.0)
        def seed_rand():
            return random.randint(0, 255)

        self.sampler = {
            'position': pos_rand,
            'look_at':  look_rand,
            'up':       up_rand,
            'radiance': rad_rand,
            'seed':     seed_rand
        }

        self.materials = [
            None,
            pyredner.Material(diffuse_reflectance = torch.tensor(
                [0.0, 0.0, 0.0], device=self.device)),
        ]

        # TODO: Revise this mesh loader once performance issue is solved
        #self.meshes = get_children_path_list(meshes_path)
        self.meshes = list(map(
            lambda mesh: pyredner.Shape(*pyredner.load_obj(mesh), 0),
            get_children_path_list(meshes_path)
        ))

        # TODO: GPU device choice is super sketchy
        self.shapes = [
            self.meshes[0],
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

    def __call__(self, sample):
        # Set Material
        # TODO: assert sample is of type Tensor on the correct device
        self.materials[0] = pyredner.Material(
                diffuse_reflectance = sample,
                specular_reflectance = torch.tensor(
                    [0.7, 0.7, 0.7], device=self.device),
                )
        # Sample mesh choice
        # TODO: performance boost required for loading many meshes
        self.shapes[0] = random.choice(self.meshes)

        # Sample Camera and Light Params
        camera = pyredner.Camera(
            position     = torch.tensor(self.sampler['position']()),
            look_at      = torch.tensor(self.sampler['look_at']()),
            up           = torch.tensor(self.sampler['up']()),
            fov          = torch.tensor([self.fov]),
            clip_near    = self.clip_near,
            resolution   = self.resolution)
        lights = [pyredner.Light(1, torch.tensor(self.sampler['radiance']()))]

        # Serialize Scene
        scene = pyredner.Scene(
            camera, self.shapes, self.materials, lights)
        args = pyredner.RenderFunction.serialize_scene(
            scene = scene, num_samples = self.num_samples, max_bounces = 1)
        return pyredner.RenderFunction.apply(self.sampler['seed'](), *args)

class MeshRenderLayer(nn.Module):
    def __init__(self, mesh_path, out_sz, num_samples, device):
        super(MeshRenderLayer, self).__init__()
        self.renderer = MeshRenderer(mesh_path, out_sz, num_samples, device)

    def forward(self, input):
        out = self.renderer(input)
        return out

class HWC2CHWLayer(nn.Module):
    def __init__(self):
        super(HWC2CHWLayer, self).__init__()
        self.transform = HWC2CHW()

    def forward(self, input):
        return self.transform(input)

class CHW2HWCLayer(nn.Module):
    def __init__(self):
        super(CHW2HWCLayer, self).__init__()
        self.transform = CHW2HWC()

    def forward(self, input):
        return self.transform(input)

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
            #bkgd_color = (random.uniform(0.0, 1.0),
            #              random.uniform(0.0, 1.0),
            #              random.uniform(0.0, 1.0))
            bkgd_color = (0.8, 0.8, 0.8)
            self.bkgd = torch.tensor(bkgd_color, device=self.device)\
                             .expand(self.size, self.size, len(bkgd_color))
        alpha = input[:,:,-1:]
        return alpha * input[:,:,:-1] + (1 - alpha) * self.bkgd

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.transform = Normalize(mean, std)

    def forward(self, input):
        return self.transform(input)

class StripBatchDimLayer(nn.Module):
    def __init__(self):
        super(StripBatchDimLayer, self).__init__()
        self.transform = StripBatchDim()

    def forward(self, input):
        return self.transform(input)

class AddBatchDimLayer(nn.Module):
    def __init__(self):
        super(AddBatchDimLayer, self).__init__()
        self.transform = AddBatchDim()

    def forward(self, input):
        return self.transform(input)

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
