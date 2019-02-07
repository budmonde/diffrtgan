import os
import math
import random

import numpy as np

import torch

import redner
import pyredner

from .image_util import imread
from .transform_util import CornerCrop


class Render(object):
    def __init__(self, meshes_path, out_sz, num_samples, max_bounces, device):
        super(Render, self).__init__()
        # Redner GPU Configs
        self.device = device
        pyredner.set_use_gpu(self.device != torch.device('cpu'))
        pyredner.set_device(self.device)

        # Render output Configs
        self.resolution = (out_sz, out_sz)
        self.num_samples = num_samples
        self.max_bounces = max_bounces

        # Camera Config Sampling
        def pos_sample():
            #d = 3.0 + random.uniform(1.0, 1.0) * 3.0
            #phi = random.uniform(0.25, 0.25) * math.pi * 2
            #theta = math.acos(1 - random.uniform(1.0, 1.0))
            #x, y, z = math.cos(phi)*math.sin(theta)*d,\
            #          math.cos(theta)*d,\
            #          math.sin(phi)*math.sin(theta)*d
            #return (x, y, z)
            return (0.0, 0.75, random.choice([-8.0, 8.0]))
        def look_sample():
            #x, y, z = random.uniform(-1.0, 1.0),\
            #          random.uniform(-1.0, 1.0,\
            #          random.uniform(-1.0, 1.0)
            #return (x, y, z)
            return (0.0, 0.75, 0.0)
        def up_sample():
            #x, y, z = random.uniform(-0.2, 0.2),\
            #          1.0,\
            #          random.uniform(-0.2, 0.2)
            #return (x, y, z)
            return (0.0, 1.0, 0.0)
        def seed_sample():
            return random.randint(0, 255)

        self.sampler = {
            'position': pos_sample,
            'look_at':  look_sample,
            'up':       up_sample,
            'seed':     seed_sample
        }

        # Load Car Meshes
        def get_children_path_list(path):
            return ['{}/{}'.format(path, f) for f in os.listdir(path)]

        def get_mesh_geometry(mesh_path):
            mtl, geo, light = pyredner.load_obj(mesh_path)
            main = geo[0][1]
            return {
                'vertices': main.vertices,
                'indices' : main.indices,
                'uvs'     : main.uvs,
                'normals' : main.normals
            }

        def define_Shape(kwargs, m_id):
            return pyredner.Shape(**kwargs, material_id = m_id)

        self.meshes = list(map(
            lambda mesh_path: define_Shape(get_mesh_geometry(mesh_path), 0),
            get_children_path_list(meshes_path)
        ))

        # NOTE: temporarily disabled until shape id feature is enabled
        # Load floor mesh and material
        #self.floor_shape = define_Shape(
        #    get_mesh_geometry('datasets/meshes/plane/plane.obj'), 1)

        #self.floor_mtl = pyredner.Material(
        #    diffuse_reflectance = torch.tensor(
        #        [0.5, 0.5, 0.5], device = self.device)
        #)

        # Load Environment Map
        # NOTE: Tzu-mao has a bug in his code in texture.py:41 for large env
        img = torch.tensor(imread('datasets/envmaps/sunsky.exr'),\
            dtype=torch.float32, device=self.device)
        self.envmap = pyredner.EnvironmentMap(img)

    def __call__(self, sample):
        # Sample Material
        assert(isinstance(sample, torch.Tensor))
        assert(sample.device == self.device)
        material = pyredner.Material(
            diffuse_reflectance = sample,
            specular_reflectance = torch.tensor(
                [0.7, 0.7, 0.7], device=self.device),
            roughness = torch.tensor(
                [0.05], device=self.device)
        )

        # Sample mesh choice
        shape = random.choice(self.meshes)

        # Sample Camera Params
        camera = pyredner.Camera(
            position     = torch.tensor(self.sampler['position']()),
            look_at      = torch.tensor(self.sampler['look_at']()),
            up           = torch.tensor(self.sampler['up']()),
            fov          = torch.tensor([45.0]),
            clip_near    = 1e-2,
            resolution   = self.resolution,
            fisheye      = False)

        # Serialize Scene
        # IMPORTANT: saving scene to the object.
        # prevents python garbage collection from
        # removing variables redner allocates
        self.scene = pyredner.Scene(
            camera,
            [shape],#self.floor_shape],
            [material],#self.floor_mtl],
            [], self.envmap)
        args = pyredner.RenderFunction.serialize_scene(
            scene = self.scene,
            num_samples = self.num_samples,
            max_bounces = self.max_bounces,
            channels = [redner.channels.radiance, redner.channels.alpha])
        out = pyredner.RenderFunction.apply(self.sampler['seed'](), *args)

        black = torch.zeros((*self.resolution, 3), device = self.device)
        out = out[:, :, :3] * out[:, :, 3:4] + black * (1 - out[:, :, 3:4])
        return out

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
