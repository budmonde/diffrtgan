import os
import math
import random

import numpy as np

import torch

import pyredner

from .image_util import imread
from .transform_util import CornerCrop


class MeshRender(object):
    def __init__(self, meshes_path, out_sz, num_samples, device):
        super(MeshRender, self).__init__()
        self.fov = 45.0
        self.clip_near = 0.01
        self.resolution = (out_sz, out_sz)
        self.num_samples = num_samples
        self.device = device

        def pos_sample():
            #d = 3.0 + random.uniform(1.0, 1.0) * 3.0
            #phi = random.uniform(0.25, 0.25) * math.pi * 2
            #theta = math.acos(1 - random.uniform(1.0, 1.0))
            #x, y, z = math.cos(phi)*math.sin(theta)*d,\
            #          math.cos(theta)*d,\
            #          math.sin(phi)*math.sin(theta)*d
            #return (x, y, z)
            return (0.0, 1.5, random.choice([-9.0, 9.0]))
        def look_sample():
            #x, y, z = random.uniform(-1.0, 1.0),\
            #          random.uniform(-1.0, 1.0,\
            #          random.uniform(-1.0, 1.0)
            #return (x, y, z)
            return (0.0, 1.5, 0.0)
        def up_sample():
            #x, y, z = random.uniform(-0.2, 0.2),\
            #          1.0,\
            #          random.uniform(-0.2, 0.2)
            #return (x, y, z)
            return (0.0, 1.0, 0.0)
        def rad_sample():
            return (2.0, 2.0, 2.0)
        def seed_sample():
            return random.randint(0, 255)

        self.sampler = {
            'position': pos_sample,
            'look_at':  look_sample,
            'up':       up_sample,
            'seed':     seed_sample
        }

        # TODO: Revise this mesh loader once performance issue is solved
        #self.meshes = get_children_path_list(meshes_path)
        def get_children_path_list(path):
            return ['{}/{}'.format(path, f) for f in os.listdir(path)]

        # TODO: Allow for legit meshes to be loaded
        def get_mesh_geometry(mesh_path):
            mtl, geo, light = pyredner.load_obj(mesh_path)
            #TODO: this should be extended further later
            main = geo[0][1]
            return main.vertices, main.indices, main.uvs, main.normals

        self.meshes = list(map(
            lambda mesh_path: pyredner.Shape(*get_mesh_geometry(mesh_path), 0),
            get_children_path_list(meshes_path)
        ))

        # NOTE: Tzu-mao has a bug in his code in texture.py:41 for large env
        img = torch.tensor(imread('datasets/envmaps/sunsky.exr'), dtype=torch.float32, device=self.device)
        self.envmap = pyredner.EnvironmentMap(img)

    def __call__(self, sample):
        # Sample Material
        assert(isinstance(sample, torch.Tensor))
        material = pyredner.Material(
                diffuse_reflectance = sample,
                specular_reflectance = torch.tensor(
                    [0.7, 0.7, 0.7], device=self.device),
                )
        # Sample mesh choice
        shape = random.choice(self.meshes)

        # Sample Camera Params
        camera = pyredner.Camera(
            position     = torch.tensor(self.sampler['position']()),
            look_at      = torch.tensor(self.sampler['look_at']()),
            up           = torch.tensor(self.sampler['up']()),
            fov          = torch.tensor([self.fov]),
            clip_near    = self.clip_near,
            resolution   = self.resolution)

        # Serialize Scene
        scene = pyredner.Scene(
            camera, [shape], [material], [], self.envmap)
        args = pyredner.RenderFunction.serialize_scene(
            scene = scene, num_samples = self.num_samples, max_bounces = 1)
        return pyredner.RenderFunction.apply(self.sampler['seed'](), *args)

class Composit(object):
    def __init__(self, bkgd, size, device):
        super(Composit, self).__init__()
        self.size = size
        self.device = device
        self.crop = CornerCrop(size)
        # TODO: this is probably not the best way to do this
        self.rng = bkgd is None

        if self.rng:
            self.bkgd = None
        else:
            self.bkgd = torch.tensor(self.crop(bkgd), device=device)

    def __call__(self, input):
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
