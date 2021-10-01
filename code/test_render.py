import torch

from util.image_util import imread, imwrite
from util.misc_util import *
from util.render_util import RenderConfig
from util.torch_util import RenderLayer


device = torch.device('cuda:0')
path = "./datasets/textures/dirt/octavia_clean.png"

scene_dict = {
    'test': {
        'cam_rotation'      : [   2.5,  0.15,   0.0],
        'cam_translation'   : [   0.0, -0.75,   0.0],
        'cam_distance'      :     7.0,
        'cam_fov'           : [  45.0],
        'cam_resolution'    : [   512,   512],
        'geometry_path'     : './datasets/meshes/serialized/octavia_clean.pth',
        'tex_diffuse_color' : [   0.8,   0.8,   0.8],
        'tex_specular_color': [   0.8,   0.8,   0.8],
        'envmap_path'       : './datasets/envmaps/one/sunsky.exr',
        'envmap_signal_mean':     0.5,
        'envmap_rotation'   :     0.0,
        'opt_num_samples'   : [   200,     1],
        'opt_max_bounces'   :       2,
        'opt_channels_str'  : ['radiance'],
        'opt_render_seed'   :       0,
    }
}
render_config = RenderConfig()
render_layer  = RenderLayer(render_config, device)

img = torch.tensor(imread(path), dtype=torch.float32, device=device)

#for fpath in get_child_paths(meshes_path):
#    print(f'Rendering {fpath}')
#    render_config.data[render_config.cfg_id]['geo_mesh_path'] = fpath
#    name = get_fn(fpath)
#    out = render_layer(opaque)
#    imwrite(out, f'debug/new_mesh_qual/{name}.png')
render_config.set_scene(scene_dict['test'])
out = render_layer(img)
imwrite(out, "debug/test_render_out.png")
