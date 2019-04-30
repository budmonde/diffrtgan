import argparse
from datetime import datetime
import json
import os
import time

from skimage.transform import resize
import torch

from util.image_util import imread, imwrite
from util.misc_util import *
from util.render_util import Render, RenderConfig
from util.sample_util import *


def main():
    # Load arguments
    parser = argparse.ArgumentParser()
    # Scene args
    parser.add_argument('--geometry_path', type=str,
            default='./datasets/meshes/clean_serialized')
    parser.add_argument('--envmaps_path', type=str,
            default='./datasets/envmaps/one')
    parser.add_argument('--diffuse_refl_path', type=str,
            default='./datasets/distributions/diffuse.txt')
    parser.add_argument('--textures_path', type=str,
            default='./datasets/textures/curve')
    parser.add_argument('--texture_size', type=int, default=256)
    # Output args
    parser.add_argument('--root_path', type=str, default='./datasets/renders/')
    parser.add_argument('--num_imgs', type=int, default=1000)
    parser.add_argument('--label', type=str, default='debug')
    # Misc
    parser.add_argument('--gpu_id', type=int, default=0,)
    opt = parser.parse_args()

    # Create Output directory
    now = datetime.now()
    subdir = f'{opt.label}_{now.month}-{now.day}-{now.hour}-{now.minute}'
    out_path = os.path.join(opt.root_path, subdir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Load samplers
    sampler = ConfigSampler({
        'cam_rotation'       : HemisphereSamplerFactory(
                                 [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]),
        'cam_translation'    : BoxSamplerFactory(
                                 [-0.1, 0.1], [-0.76, -0.74], [-0.1, 0.1]),
        'cam_distance'       : ConstantSamplerFactory(7.0),
        'cam_fov'            : ConstantSamplerFactory([45.0]),
        'cam_resolution'     : ConstantSamplerFactory([256, 256]),
        'geometry_path'      : PathSamplerFactory(opt.geometry_path, ext='pth'),
        'tex_diffuse_color'  : RGBFileSamplerFactory(opt.diffuse_refl_path),
        'tex_specular_color' : ConstantSamplerFactory([0.8, 0.8, 0.8]),
        'envmap_path'        : PathSamplerFactory(opt.envmaps_path, ext='exr'),
        'envmap_signal_mean' : ConstantSamplerFactory(0.5),
        'envmap_rotation'    : ConstantSamplerFactory(0.0),
        'opt_num_samples'    : ConstantSamplerFactory((200, 1)),
        'opt_max_bounces'    : ConstantSamplerFactory(2),
        'opt_channels_str'   : ConstantSamplerFactory(['radiance', 'alpha']),
        'opt_render_seed'    : RandIntSamplerFactory(0, 1e6),
    })

    # Init renderer
    device = torch.device(f'cuda:{opt.gpu_id}' if opt.gpu_id != -1 else 'cpu')
    config = RenderConfig()
    renderer = Render(config, device)

    log = dict()
    for i in range(opt.num_imgs):
        # Generate render id and scene configs
        key = gen_hash(6)
        while key in log.keys():
            key = gen_hash()
        scene = sampler.generate()
        log[key] = scene
        config.set_scene(scene)

        # Set texture for rendering
        mesh_name = get_fn(config('geometry_path'))
        texture = imread(os.path.join(opt.textures_path, f'{mesh_name}.png'))
        texture = resize(texture, (opt.texture_size, opt.texture_size))
        texture = torch.tensor(texture, dtype=torch.float32, device=device)

        # Time Render operation
        iter_start_time = time.time()
        out = renderer(texture)
        render_time = time.time() - iter_start_time
        print(f'Generated Image: #\t{i} -- {key} in {render_time}')
        imwrite(out[...,  : 3], os.path.join(out_path, 'img',  f'{key}.png'))
        imwrite(out[..., 3: 4], os.path.join(out_path, 'mask', f'{key}.png'))

        with open(os.path.join(out_path, 'data.json'), 'w') as metafile:
            json.dump(log, metafile)

if __name__ == '__main__':
    main()
