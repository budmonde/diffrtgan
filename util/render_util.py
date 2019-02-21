import os
import math
import random
import hashlib

import numpy as np

import torch

import redner
import pyredner

from .image_util import imread
from .transform_util import CornerCrop


class RenderLogger(object):
    def __init__(self, data = None):
        super(RenderLogger, self).__init__()
        if data == None:
            self.data = dict()
        else:
            self.data = data
        self.latest_cfg_id = None

    def new(self, cfg_id):
        self.latest_cfg_id = cfg_id
        self.data[cfg_id] = dict()

    def log(self, key, val):
        assert(self.latest_cfg_id != None)
        self.data[self.latest_cfg_id][key] = val

    def get_data(self):
        return self.data.copy()

class RenderConfig(object):
    def __init__(self, data):
        super(RenderConfig, self).__init__()
        self.data = data

    def set_config(self, cfg_id):
        self.cfg_id = cfg_id

    def get_val(self, key):
        return self.data[self.cfg_id][key]

class Render(object):
    def __init__(self, meshes_path, envmaps_path, out_sz, num_samples, max_bounces, device, channels = ['radiance', 'alpha'], logger = None, config = None):
        super(Render, self).__init__()
        # Image write logger and scene config overrider
        self.logger = logger
        self.config = config
        self.sampler = dict()

        # Redner GPU Configs
        self.device = device
        pyredner.set_use_gpu(self.device != torch.device('cpu'))
        pyredner.set_device(self.device)

        # Render output configs
        self.resolution = (out_sz, out_sz)
        self.num_samples = num_samples
        self.max_bounces = max_bounces

        self.channels = list()
        for ch in channels:
            self.channels.append(getattr(redner.channels, ch))

        def get_children_path_list(path):
            return [os.path.join(path, f) for f in os.listdir(path)]

        # Load Car Meshes
        def get_mesh_geometry(mesh_path):
            mtl_map, mesh_list, _ = pyredner.load_obj(mesh_path)
            mtl_id_map = dict()
            materials = list()
            cnt = 0
            for k, v in mtl_map.items():
                mtl_id_map[k] = cnt
                cnt += 1
                materials.append(v)
            # TODO: hardcoded. fix to make it load from a config file
            learn_tex_idx = mtl_id_map['car_paint']
            shapes = list()
            for mtl_name, mesh in mesh_list:
                shapes.append(pyredner.Shape(
                    vertices = mesh.vertices,
                    indices = mesh.indices,
                    uvs = mesh.uvs,
                    normals = mesh.normals,
                    material_id = mtl_id_map[mtl_name]
                ))
            return {
                'materials': materials,
                'shapes': shapes,
                'learn_tex_idx': learn_tex_idx,
            }

        self.meshes = dict(map(
            lambda mesh_path: (mesh_path, get_mesh_geometry(mesh_path)),
            list(filter(
                lambda fn: fn.split('.')[-1] == 'obj',
                get_children_path_list(meshes_path)
            ))
        ))

        def geo_mesh_path_sample(override = None, logger = None):
            if override == None:
                key = random.choice(list(self.meshes.keys()))
            else:
                key = override
            if logger != None:
                logger.log('geo_mesh_path', key)
            return key
        self.sampler['geo_mesh_path'] = geo_mesh_path_sample

        def get_envmap(envmap_path):
            return pyredner.EnvironmentMap(torch.tensor(imread(envmap_path),\
                dtype=torch.float32, device=self.device))

        # Load Environment Maps
        self.envmaps = dict(map(
            lambda envmap_path: (envmap_path, get_envmap(envmap_path)),
            get_children_path_list(envmaps_path)
        ))

        def geo_envmap_path_sample(override = None, logger = None):
            if override == None:
                key = random.choice(list(self.envmaps.keys()))
            else:
                key = override
            if logger != None:
                logger.log('geo_envmap_path', key)
            return key
        self.sampler['geo_envmap_path'] = geo_envmap_path_sample

        # Camera Config Sampling
        def cam_position_sample(override = None, logger = None):
            if override == None:
                d = 6.0 + random.uniform(1.0, 1.0) * 1.0
                phi = random.uniform(0.24, 0.25) * math.pi * 2
                theta = math.acos(1 - random.uniform(0.95, 1.0))
                x, y, z = math.cos(phi)*math.sin(theta)*d,\
                          math.cos(theta)*d + 0.75,\
                          math.sin(phi)*math.sin(theta)*d
                out = (x, y, z)
            else:
                out = override
            #out = (0.0, 0.75, random.choice([-8.0, 8.0]))
            if logger != None:
                logger.log('cam_position', out)
            return out
        def cam_look_at_sample(override = None, logger = None):
            if override == None:
                x, y, z = random.uniform(-0.1, 0.1),\
                          random.uniform(-0.1, 0.1) + 0.75,\
                          random.uniform(-0.1, 0.1)
                out = (x, y, z)
            else:
                out = override
            #out = (0.0, 0.75, 0.0)
            if logger != None:
                logger.log('cam_look_at', out)
            return out
        def cam_up_sample(override = None, logger = None):
            #x, y, z = random.uniform(-0.2, 0.2),\
            #          1.0,\
            #          random.uniform(-0.2, 0.2)
            if override == None:
                out =  (0.0, 1.0, 0.0)
            else:
                out = override
            if logger != None:
                logger.log('cam_up', out)
            return out
        def render_seed_sample(override = None, logger = None):
            if override == None:
                out = random.randint(0, 255)
            else:
                out = override
            if logger != None:
                logger.log('render_seed', out)
            return out
        self.sampler['cam_position'] = cam_position_sample
        self.sampler['cam_look_at'] = cam_look_at_sample
        self.sampler['cam_up'] = cam_up_sample
        self.sampler['render_seed'] = render_seed_sample

    def __call__(self, input, name = None):
        # Init Logger entry
        if self.logger != None:
            if name == None:
                name = hashlib.sha256(str(random.randint(0, 10000)).encode('utf-8')).hexdigest()[:6]
            self.logger.new(name)

        # Init Configurator
        if self.config == None:
            mesh_path = self.sampler['geo_mesh_path'](logger = self.logger)
            envmap_path = self.sampler['geo_envmap_path'](logger = self.logger)
            position = self.sampler['cam_position'](logger = self.logger)
            look_at = self.sampler['cam_look_at'](logger = self.logger)
            up = self.sampler['cam_up'](logger = self.logger)
            seed = self.sampler['render_seed'](logger = self.logger)
        else:
            mesh_path = self.sampler['geo_mesh_path'](self.config.get_val('geo_mesh_path'), logger = self.logger)
            envmap_path = self.sampler['geo_envmap_path'](self.config.get_val('geo_envmap_path'), logger = self.logger)
            position = self.sampler['cam_position'](self.config.get_val('cam_position'), logger = self.logger)
            look_at = self.sampler['cam_look_at'](self.config.get_val('cam_look_at'), logger = self.logger)
            up = self.sampler['cam_up'](self.config.get_val('cam_up'), logger = self.logger)
            seed = self.sampler['render_seed'](self.config.get_val('render_seed'), logger = self.logger)

        # Sample car_mesh Choice
        car_mesh = self.meshes[mesh_path]
        shapes = car_mesh['shapes']
        materials = car_mesh['materials']

        # Set Learneable Material
        assert(isinstance(input, torch.Tensor))
        assert(input.device == self.device)
        assert(input.shape[-1] == 3 or input.shape[-1] == 4)

        # Alpha composit if input is semi-transparent
        if input.shape[-1] == 4:
            diffuse_background_color = (0.8, 0.8, 0.8)
            specular_background_color = (0.8, 0.8, 0.8)
            diffuse_background = torch.tensor(diffuse_background_color, device=self.device)\
                                      .expand(*input.shape[:2], len(diffuse_background_color))
            specular_background = torch.tensor(specular_background_color, device=self.device)\
                                       .expand(*input.shape[:2], len(specular_background_color))
            diffuse_reflectance = input[:,:,-1:] * input[:,:,:-1] + (1 - input[:,:,-1:]) * diffuse_background
            specular_reflectance = (1 - input[:,:,-1:]) * specular_background
        else:
            diffuse_reflectance = input
            specular_reflectance = torch.tensor((0.8, 0.8, 0.8), device=self.device)

        materials[car_mesh['learn_tex_idx']] = pyredner.Material(
            diffuse_reflectance = diffuse_reflectance,
            specular_reflectance = specular_reflectance,
            roughness = torch.tensor([0.05], device=self.device)
        )

        # Sample environment map choice
        envmap = self.envmaps[envmap_path]

        # Sample Camera Params
        camera = pyredner.Camera(
            position     = torch.tensor(position),
            look_at      = torch.tensor(look_at),
            up           = torch.tensor(up),
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
            shapes,
            materials,
            [], envmap)
        args = pyredner.RenderFunction.serialize_scene(
            scene = self.scene,
            num_samples = self.num_samples,
            max_bounces = self.max_bounces,
            channels = self.channels)
        out = pyredner.RenderFunction.apply(seed, *args)

        return out

# TODO: this is not the best workaround for my new Renderer
class PostComposit(object):
    def __init__(self, size, device):
        super(PostComposit, self).__init__()
        self.resolution = (size, size)
        self.device = device

    def __call__(self, input):
        black = torch.zeros((*self.resolution, 3), device = self.device)
        return input[:, :, :3] * input[:, :, 3:4] + black * (1 - input[:, :, 3:4])

class Composit(object):
    def __init__(self, background, size, device):
        super(Composit, self).__init__()
        self.size = size
        self.device = device
        self.crop = CornerCrop(size)
        # TODO: this is probably not the best way to do this
        self.rng = background is None

        if self.rng:
            self.background = None
        else:
            self.background = torch.tensor(self.crop(background), device=device)

    def __call__(self, input):
        # input is in format HWC
        if self.rng:
            #background_color = (random.uniform(0.0, 1.0),
            #              random.uniform(0.0, 1.0),
            #              random.uniform(0.0, 1.0))
            background_color = (0.8, 0.8, 0.8)
            self.background = torch.tensor(background_color, device=self.device)\
                             .expand(self.size, self.size, len(background_color))
        alpha = input[:,:,-1:]
        return alpha * input[:,:,:-1] + (1 - alpha) * self.background
