import argparse
import json
import math
import os

import torch

import redner
import pyredner

from util.geometry_util import camera_parameters
from util.image_util import imread, imwrite
from util.misc_util import *


# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--models_path', type=str, default='./datasets/meshes/one', help='Model path for mesh to render')
parser.add_argument('--resolution', type=int, default=256, help='Resolution')
parser.add_argument('--num_elev', type=int, default=50, help='Number of samples along azimuth parameter')
parser.add_argument('--min_elev', type=float, default=0.0, help='Minimum elevation')
parser.add_argument('--max_elev', type=float, default=math.pi/8.0, help='Maximum elevation')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')
opt = parser.parse_args()


def generate_poses(model_path, output_path):
    # Init logger
    log = dict()

    # Load renderer configs
    material_map, mesh_list, light_map = pyredner.load_obj(model_path)
    material_id_map = {}
    materials = []
    count = 0
    for key, value in material_map.items():
        material_id_map[key] = count
        count += 1
        materials.append(value)

    shapes = []
    for mtl_name, mesh in mesh_list:
        shapes.append(
            pyredner.Shape(
                vertices = mesh.vertices,
                indices = mesh.indices,
                uvs = mesh.uvs,
                normals = mesh.normals,
                material_id = material_id_map[mtl_name]
            )
        )

    envmap = pyredner.EnvironmentMap(torch.tensor(
        imread('./datasets/envmaps/one/sunsky.exr'),
        dtype=torch.float32,
        device=pyredner.get_device()))

    # Object pose parameters
    euler_angles = [0.0, 0.0, 0.0]
    translation = [0.0, -0.75, 0.0]
    up = [0.0, 1.0, 0.0]
    distance = 7.0

    # Setup base scene to modify during iterations
    cam_params = camera_parameters(euler_angles, translation, distance, up)

    camera = pyredner.Camera(
            position = torch.tensor(cam_params[0], dtype=torch.float32),
            look_at = torch.tensor(cam_params[1], dtype=torch.float32),
            up = torch.tensor(cam_params[2], dtype=torch.float32),
            fov = torch.tensor([45.0]),
            clip_near = 1e-2,
            resolution = (opt.resolution, opt.resolution),
            fisheye = False)

    scene = pyredner.Scene(camera, shapes, materials,
            area_lights = [], envmap = envmap)

    # Generate alphamasks
    for i in range(opt.num_elev):
        # Set elevation angle
        elev_pc = i / opt.num_elev
        elevation = opt.max_elev * elev_pc + opt.min_elev * (1 - elev_pc)
        euler_angles[1] = elevation

        # Calculate number of azimuthal iterations
        num_azimuth = int(opt.num_elev * math.sin(math.pi/2 - elevation))
        for j in range(num_azimuth):
            # Set azimuthal angle
            azimuth_pc = j / num_azimuth
            azimuth = math.pi * 2 * azimuth_pc

            euler_angles[0] = azimuth

            print('Params: Elevation - {:.4f}\tAzimuth - {:.4f}'\
                    .format(elevation, azimuth))

            # Set Camera params
            cam_params = camera_parameters(euler_angles, translation, distance, up)

            # Update scene params
            scene.camera = pyredner.Camera(
                position = torch.tensor(cam_params[0], dtype=torch.float32),
                look_at = torch.tensor(cam_params[1], dtype=torch.float32),
                up = torch.tensor(cam_params[2], dtype=torch.float32),
                fov = torch.tensor([45.0]),
                clip_near = 1e-2,
                resolution = (opt.resolution, opt.resolution),
                fisheye = False
            )
            args = pyredner.RenderFunction.serialize_scene(
                scene = scene,
                num_samples = 1,
                max_bounces = 1,
                channels = [redner.channels.alpha])

            out = pyredner.RenderFunction.apply(1, *args)

            fn = gen_hash(6)
            imwrite(out, os.path.join(output_path, '{}.png'.format(fn)))
            log[fn] = {
                'elevation': elevation,
                'azimuth': azimuth
            }
    return log

# Set Redner device
device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id != -1 else 'cpu')
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_device(device)

# Generate poses
paths = get_child_paths(opt.models_path)
paths = list(filter(lambda p: get_ext(p) == 'obj', paths))

for path in paths:
    print('Generating poses for {}'.format(path))

    model_name = get_fn(path)
    poses_dir = os.path.join('./datasets/poses', model_name)

    if not os.path.exists(poses_dir):
        os.makedirs(poses_dir)

    log = generate_poses(path, poses_dir)
    with open(os.path.join(poses_dir, 'data.json'), 'w') as f:
        json.dump(log, f)
