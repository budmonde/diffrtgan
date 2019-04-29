import os
import math
import random

import numpy as np

import torch

import redner
import pyredner

from .geometry_util import camera_parameters, get_rotation_matrix_y
from .image_util import imread
from .misc_util import *


class RenderConfig(object):
    def __init__(self):
        super(RenderConfig, self).__init__()
        self.scene = None

    def set_scene(self, scene_dict):
        self.scene = scene_dict
        return True

    def __call__(self, key):
        assert(self.scene is not None)
        assert(key in self.scene)
        return self.scene[key]


class Render(object):
    def __init__(self, config, device):
        super(Render, self).__init__()
        # Initialize scene metadata config
        self.config = config

        # Redner GPU Configs
        self.device = device
        pyredner.set_use_gpu(self.device != torch.device('cpu'))
        pyredner.set_device(self.device)

    def __call__(self, input):
        ### Load active scene blob from config ###

        # Camera params
        cam_rotation           = self.config('cam_rotation')
        cam_translation        = self.config('cam_translation')
        cam_distance           = self.config('cam_distance')
        cam_fov                = self.config('cam_fov')
        cam_resolution         = self.config('cam_resolution')
        # Geometry params
        geometry_path          = self.config('geometry_path')
        # Texture params
        tex_diffuse_color      = self.config('tex_diffuse_color')
        tex_specular_color     = self.config('tex_specular_color')
        # Envmap params
        envmap_path            = self.config('envmap_path')
        envmap_signal_mean     = self.config('envmap_signal_mean')
        envmap_rotation        = self.config('envmap_rotation')
        # Render option params
        opt_num_samples        = self.config('opt_num_samples')
        opt_max_bounces        = self.config('opt_max_bounces')
        opt_channels_str       = self.config('opt_channels_str')
        opt_render_seed        = self.config('opt_render_seed')

        # XXX: Temporary hack for training override
        opt_num_samples = (200, 1)
        opt_channels_str = ['radiance']

        ### Load configs as pyredner primitives ###

        # Convert Camera params for pyredner.Camera object
        position, look_at, up = camera_parameters(
                cam_rotation, cam_translation, cam_distance)
        camera = pyredner.Camera(
                position     = torch.tensor(position, dtype=torch.float32),
                look_at      = torch.tensor(look_at, dtype=torch.float32),
                up           = torch.tensor(up, dtype=torch.float32),
                fov          = torch.tensor(cam_fov),
                clip_near    = 1e-2,  # Hardcoded
                resolution   = cam_resolution,
                fisheye      = False) # Hardcoded

        # Load geometry from specified path
        mesh = LearnMesh.load_pth(geometry_path, self.device)

        # Set Learnable material
        mesh.set_learn_material(input, tex_diffuse_color, tex_specular_color)

        # Load envmap from specified path
        envmap = load_envmap(
                envmap_path,
                envmap_signal_mean,
                envmap_rotation,
                self.device)

        # Convert channels list into redner primitives
        opt_channels = list()
        for ch in opt_channels_str:
            opt_channels.append(getattr(redner.channels, ch))

        # Serialize Scene
        # IMPORTANT: saving scene to the object.
        # prevents python garbage collection from
        # removing variables redner allocates
        self.scene = pyredner.Scene(
                camera,
                mesh.shapes,
                mesh.materials,
                [], envmap)
        # XXX: Temporary hack
        args = pyredner.RenderFunction.serialize_scene(
                scene = self.scene,
                num_samples = opt_num_samples,
                max_bounces = opt_max_bounces,
                channels = opt_channels)
        render = pyredner.RenderFunction.apply(opt_render_seed, *args)

        #TODO: Add flag to ask whether to normalize
        #render /= torch.mean(render) * 2
        out = torch.clamp(render, 0, 1)

        return out

class LearnMesh(object):
    def __init__(self, materials, shapes, learn_tex_idx, device):
        self.materials = materials
        self.shapes = shapes
        self.learn_tex_idx = learn_tex_idx
        self.device = device
        self.ground_idx = None

    def set_learn_material(self, input, diffuse, specular):
        assert(isinstance(input, torch.Tensor))
        assert(input.device == self.device)
        assert(input.shape[-1] == 3 or input.shape[-1] == 4)

        # Alpha composit learneable material if input is semi-transparent
        if input.shape[-1] == 4:
            diffuse_torch  = torch.tensor(diffuse, device=self.device)\
                                  .expand(*input.shape[:2], len(diffuse))
            specular_torch = torch.tensor(specular, device=self.device)\
                                  .expand(*input.shape[:2], len(specular))
            diffuse_torch  =      input[:,:,-1:]  * input[:,:,:-1] + \
                             (1 - input[:,:,-1:]) * diffuse_torch
            specular_torch = (1 - input[:,:,-1:]) * specular_torch
        else:
            diffuse_torch  = input
            specular_torch = torch.tensor(specular, device=self.device)

        self.materials[self.learn_tex_idx] = pyredner.Material(
                diffuse_reflectance = diffuse_torch,
                specular_reflectance = specular_torch,
                roughness = torch.tensor([0.05], device=self.device))

    def add_ground(self, device):
        self.materials.append(pyredner.Material(
                diffuse_reflectance = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32, device=device),
                specular_reflectance = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
                roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
        ))
        self.shapes.append(pyredner.Shape(
                vertices = torch.tensor([[-100, 0, -100],[-100, 0, 100],[100,0,-100],[100,0,100]], dtype=torch.float32, device=device),
                indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device=device),
                normals = torch.tensor([[0,1,0],[0,1,0],[0,1,0]], dtype=torch.float32, device=device),
                uvs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32, device=device),
                material_id=len(self.materials)-1,
        ))
        self.ground_idx = (len(self.materials)-1, len(self.shapes)-1)

    def remove_ground(self):
        if self.ground_idx != None:
            del self.materisla[ground_idx[0]]
            del self.shapes[ground_idx[1]]

    def state_dict(self):
        return {
            'materials': [m.state_dict() for m in self.materials],
            'shapes': [s.state_dict() for s in self.shapes],
            'learn_tex_idx': self.learn_tex_idx
        }

    @classmethod
    def load_obj(cls, path, learn_tex_label, device):
        mtl_map, mesh_list, _ = pyredner.load_obj(path)
        mtl_id_map = dict()
        materials = list()
        cnt = 0
        for k, v in mtl_map.items():
            mtl_id_map[k] = cnt
            cnt += 1
            materials.append(v)
        assert(learn_tex_label in mtl_id_map)
        learn_tex_idx = mtl_id_map[learn_tex_label]
        shapes = list()
        for mtl_name, mesh in mesh_list:
            shapes.append(pyredner.Shape(
                vertices = mesh.vertices,
                indices = mesh.indices,
                uvs = mesh.uvs,
                normals = mesh.normals,
                material_id = mtl_id_map[mtl_name]
            ))
        return cls(materials, shapes, learn_tex_idx)

    @classmethod
    def load_state_dict(cls, state_dict, device):
        materials = [
            pyredner.Material.load_state_dict(m)
            for m in state_dict['materials']
        ]
        shapes =  [
            pyredner.Shape.load_state_dict(s)
            for s in state_dict['shapes']
        ]
        learn_tex_idx = state_dict['learn_tex_idx']

        return cls(materials, shapes, learn_tex_idx, device)

    @classmethod
    def load_pth(cls, path, device):
        state_dict = torch.load(path, map_location=device)
        return cls.load_state_dict(state_dict, device)


# Envmap Loader
def load_envmap(envmap_path, signal_mean, rangle, device):
    envmap = imread(envmap_path)
    envmap = envmap / np.mean(envmap) * signal_mean
    env_to_world = torch.tensor(get_rotation_matrix_y(rangle),
            dtype=torch.float32)
    return pyredner.EnvironmentMap(
            torch.tensor(envmap, dtype=torch.float32, device=device),
            env_to_world=env_to_world)
