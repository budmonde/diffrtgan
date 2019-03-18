import os
import math
import random
import hashlib

import numpy as np

import torch

import redner
import pyredner

from .image_util import imread


def gen_hash(length):
    return hashlib.sha256(str(random.randint(0, 1e10))\
                  .encode('utf-8')).hexdigest()[:length]

class RenderLogger(object):
    def __init__(self, data = None, write = True):
        super(RenderLogger, self).__init__()
        self.data = dict() if data is None else data
        self.write = write
        self.active_cfg_id = None

    def init(self):
        if self.write == False:
            pass
        cfg_id = gen_hash(6)
        while cfg_id in self.data:
            cfg_id = gen_hash(6)
        self.active_cfg_id = cfg_id
        self.data[cfg_id] = dict()

    def log(self, key, val):
        if self.write == False:
            pass
        assert(self.active_cfg_id != None)
        self.data[self.active_cfg_id][key] = val

    def __getitem__(self, cfg_id):
        return self.data[cfg_id]

    def get_active_id(self):
        return self.active_cfg_id

    def get_all(self):
        return self.data.copy()

class RenderConfig(object):
    def __init__(self, data=None):
        super(RenderConfig, self).__init__()
        self.data = data

    def set_config(self, cfg_id):
        # TODO: assert that this key exists in the data
        self.cfg_id = cfg_id

    def __getitem__(self, key):
        if self.data == None:
            return None
        if self.cfg_id not in self.data:
            return None
        if key not in self.data[self.cfg_id]:
            return None
        return self.data[self.cfg_id][key]

class ConfigSampler(object):
    def __init__(self, config, logger):
        super(ConfigSampler, self).__init__()
        self.config = config
        self.logger = logger
        self.samplers = dict()
        self.noises = dict()

    def add(self, fn):
        self.samplers[fn.__name__] = fn

    def add_noise(self, fn):
        self.noises[fn.__name__.strip('_noise')] = fn

    def __call__(self, key):
        if self.config[key] == None:
            val = self.samplers[key]()
        else:
            val = self.config[key]
            if key in self.noises:
                val += self.noises[key]()
        if self.logger != None:
            to_log = val
            if isinstance(val, np.ndarray):
                to_log = val.tolist()
            self.logger.log(key, to_log)
        return val

class Render(object):
    def __init__(
            self,
            meshes_path,
            envmaps_path,
            out_sz,
            num_samples,
            max_bounces,
            device,
            channels = ['radiance', 'alpha'],
            logger = RenderLogger(write = False),
            config = RenderConfig()):
        super(Render, self).__init__()
        # Image write logger and scene config overrider
        # TODO: Combine these three things(?)
        self.logger = logger
        self.config = config
        self.sampler = ConfigSampler(self.config, self.logger)

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

        # Load Meshes
        self.meshes = dict(map(
            lambda mesh_path: (mesh_path, get_mesh_geometry(mesh_path)),
            list(filter(
                lambda fn: fn.split('.')[-1] == 'obj',
                get_children_path_list(meshes_path)
            ))
        ))

        def geo_mesh_path():
            return random.choice(list(self.meshes.keys()))
        self.sampler.add(geo_mesh_path)

        # Load Environment Maps
        self.envmaps = dict(map(
            lambda envmap_path: (envmap_path, get_envmap(envmap_path, device)),
            get_children_path_list(envmaps_path)
        ))

        def tex_envmap_path():
            return random.choice(list(self.envmaps.keys()))
        self.sampler.add(tex_envmap_path)

        def geo_rotation():
            azim = random.uniform(0.00, 0.50) * math.pi * 2
            elev = math.acos(1 - random.uniform(0.00, 0.00))
            ornt = random.uniform(0.00, 0.00)
            return np.array([azim, elev, ornt])
        self.sampler.add(geo_rotation)
        def geo_rotation_noise():
            azim = random.uniform(-0.01, 0.01) * math.pi * 2
            elev = math.acos(1 - random.uniform(-0.01, 0.01))
            ornt = random.uniform(0.00, 0.00)
            return np.array([azim, elev, ornt])
        self.sampler.add_noise(geo_rotation_noise)

        def geo_translation():
            x, y, z = random.uniform(-0.1, 0.1),\
                      random.uniform(-0.1, 0.1) - 0.75,\
                      random.uniform(-0.1, 0.1)
            return np.array([x, y, z])
        self.sampler.add(geo_translation)
        def geo_translation_noise():
            x, y, z = random.uniform(-0.01, 0.01),\
                      random.uniform(-0.01, 0.01),\
                      random.uniform(-0.01, 0.01)
            return np.array([x, y, z])
        self.sampler.add_noise(geo_translation_noise)

        def geo_distance():
            return 6.0 + random.uniform(1.0, 1.0) * 1.0
        self.sampler.add(geo_distance)
        def geo_distance_noise():
            return random.uniform(-0.1, 0.1) * 1.0
        self.sampler.add_noise(geo_distance_noise)

        def render_seed():
            return random.randint(0, 255)
        self.sampler.add(render_seed)

    def __call__(self, input):
        # Init Config and Logger entry
        self.logger.init()
        geo_mesh_path   = self.sampler('geo_mesh_path')
        tex_envmap_path = self.sampler('tex_envmap_path')
        geo_rotation    = self.sampler('geo_rotation')
        geo_translation = self.sampler('geo_translation')
        geo_distance    = self.sampler('geo_distance')
        render_seed     = self.sampler('render_seed')

        # Sample car_mesh choice
        car_mesh = self.meshes[geo_mesh_path]
        shapes = car_mesh['shapes']
        materials = car_mesh['materials']

        # Set Learneable Material
        assert(isinstance(input, torch.Tensor))
        assert(input.device == self.device)
        assert(input.shape[-1] == 3 or input.shape[-1] == 4)

        # Alpha composit learneable material if input is semi-transparent
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
            roughness = torch.tensor([0.05], device=self.device))

        # Sample environment map choice
        envmap = self.envmaps[tex_envmap_path]

        # Sample Camera Params
        position, look_at, up = camera_parameters(geo_rotation, geo_translation, geo_distance)
        camera = pyredner.Camera(
            position     = torch.tensor(position, dtype=torch.float32),
            look_at      = torch.tensor(look_at, dtype=torch.float32),
            up           = torch.tensor(up, dtype=torch.float32),
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
        out = pyredner.RenderFunction.apply(render_seed, *args)

        return out

##############################
###### HELPER FUNCTIONS ######
##############################

# List of Child paths
def get_children_path_list(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

# Car Mesh Loader
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

# Envmap Loader
def get_envmap(envmap_path, device):
    return pyredner.EnvironmentMap(torch.tensor(imread(envmap_path),\
        dtype=torch.float32, device=device))

# Object transformation -> Camera Transformation
def camera_parameters(euler_angles, translation, distance, up=(0.0, 1.0, 0.0)):
    # Calculate Camera Position
    cam_azim = -euler_angles[0]
    cam_elev = math.pi/2 - euler_angles[1]
    cam_pos_hat = np.array([
        math.cos(cam_azim)*math.sin(cam_elev),
        math.cos(cam_elev),
        math.sin(cam_azim)*math.sin(cam_elev)
    ])
    cam_position = cam_pos_hat * distance - translation

    # Calculate Camera Look-at
    cam_dir = -cam_pos_hat
    cam_look_at = cam_position + cam_dir

    # Calculate Camera Up direction
    up = np.array(up)
    axis = -cam_pos_hat
    cam_up = np.dot(up, axis) * axis +\
             np.cross(axis, up) * math.sin(euler_angles[2]) +\
             np.cross(np.cross(axis, up), axis) * math.cos(euler_angles[2])
    return (cam_position, cam_look_at, cam_up)
