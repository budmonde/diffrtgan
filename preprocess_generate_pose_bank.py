import argparse
import math
import os

import torch

import redner
import pyredner

from util.image_util import imread, imwrite


# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./datasets/meshes/one/octavia_clean.obj', help='Model path for mesh to render')
parser.add_argument('--resolution', type=int, default=256, help='Resolution')
parser.add_argument('--num_azimuth', type=int, default=100, help='Number of samples along azimuth parameter')
parser.add_argument('--num_distance', type=int, default=10, help='Number of samples along distance parameter')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')
opt = parser.parse_args()


# Set Redner device
device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id != -1 else 'cpu')
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_device(device)


# Load renderer configs
# TODO: Load list of meshes
material_map, mesh_list, light_map = pyredner.load_obj(opt.model_path)
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

# TODO: move to geometry_utils
def obj2cam(euler_angles, translation, distance, up):
    # Calculate Camera Position
    cam_azim = -euler_angles[0]
    cam_elev = math.pi/2 - euler_angles[1]
    cam_pos_hat = torch.tensor([
        torch.cos(cam_azim)*torch.sin(cam_elev),
        torch.cos(cam_elev),
        torch.sin(cam_azim)*torch.sin(cam_elev)
    ])
    cam_position = cam_pos_hat * distance - translation

    # Calculate Camera Look-at
    cam_dir = -cam_pos_hat
    cam_look_at = cam_position + cam_dir

    # Calculate Camera Up direction
    axis = -cam_pos_hat
    cam_up = torch.dot(up, axis) * axis +\
             torch.cross(axis, up) * torch.sin(euler_angles[2]) +\
             torch.cross(torch.cross(axis, up), axis) * torch.cos(euler_angles[2])
    return (cam_position, cam_look_at, cam_up)

# Iteration variable
euler_angles = torch.tensor([0.0, 0.0, 0.0])
distance = torch.tensor([6.0])

# Static
translation = torch.tensor([0.0, -0.75, 0.0])
up = torch.tensor([0.0, 1.0, 0.0])

# Setup base scene to modify during iterations
cam_params = obj2cam(euler_angles, translation, distance, up)

camera = pyredner.Camera(
        position = cam_params[0],
        look_at = cam_params[1],
        up = cam_params[2],
        fov = torch.tensor([45.0]),
        clip_near = 1e-2,
        resolution = (opt.resolution, opt.resolution),
        fisheye = False)

scene = pyredner.Scene(
        camera,
        shapes,
        materials,
        area_lights = [],
        envmap = envmap)

model_name = opt.model_path.split('/')[-1].split('.')[0]
poses_dir = os.path.join('./datasets/poses', model_name, '{}_{}'.format(opt.num_azimuth, opt.num_distance))

if not os.path.exists(poses_dir):
    os.makedirs(poses_dir)

print("Generating {} poses".format(opt.num_azimuth * opt.num_distance))
for i in range(opt.num_azimuth):
    for j in range(opt.num_distance):
        print("Params: azimuth - {}\tdistance - {}".format(i, j))

        # Compute variables
        euler_angles[0] = 2 * math.pi * i / opt.num_azimuth
        distance = 6 + 4 * j / opt.num_distance

        cam_params = obj2cam(euler_angles, translation, distance, up)

        # Update scene params
        scene.camera = pyredner.Camera(
            position = cam_params[0],
            look_at = cam_params[1],
            up = cam_params[2],
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

        imwrite(out, os.path.join(poses_dir, '{}_{}.png'.format(i, j)))
