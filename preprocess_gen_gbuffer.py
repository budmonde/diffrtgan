import os
import subprocess
from tempfile import NamedTemporaryFile

from util.misc_util import *


MODELS_PATH = './datasets/meshes/full'
LEARN_MODELS_PATH = './datasets/meshes/learn'

POSITION_PATH = './datasets/gbuffers/position'
NORMAL_PATH = './datasets/gbuffers/normal'
MASK_PATH = './datasets/gbuffers/mask'

SCENE_TEMPLATE = """
LookAt 0 0 0  0 1 0  1 0 0
Camera "uv"
Film "image"
"integer xresolution" [{resolution}] "integer yresolution" [{resolution}]
    "string filename" "{output_path}.png"

Sampler "halton" "integer pixelsamples" [8]

Integrator "{integrator}"

WorldBegin
AttributeBegin
Include "{input_path}"
AttributeEnd
WorldEnd
"""

def obj2pbrt(objpath):
    model_pbrt = NamedTemporaryFile(delete=False)
    with open(os.devnull, 'w') as devnull:
        subprocess.run(['obj2pbrt', objpath, model_pbrt.name], stdout=devnull)
    return model_pbrt.name

def render_scene(input_path, output_path, resolution, integrator):
    with open(os.devnull, 'w') as devnull:
        scene_desc = SCENE_TEMPLATE.format(
                input_path = input_path,
                output_path = output_path,
                resolution = resolution,
                integrator = integrator)
        scene_pbrt = NamedTemporaryFile(delete=False)
        scene_pbrt.write(scene_desc.encode('utf-8'))
        scene_pbrt.close()
        subprocess.run(['pbrt', scene_pbrt.name], stdout=devnull)
    return

def main():
    model_paths = get_child_paths(MODELS_PATH, 'obj')
    learn_model_paths = get_child_paths(LEARN_MODELS_PATH, 'obj')

    if not os.path.exists(POSITION_PATH):
        os.makedirs(POSITION_PATH)
    if not os.path.exists(NORMAL_PATH):
        os.makedirs(NORMAL_PATH)
    if not os.path.exists(MASK_PATH):
        os.makedirs(MASK_PATH)

    for path in learn_model_paths:
        print(f"Rendering learn mask for {path}")
        # convert obj to pbrt
        learn_model_pbrt = obj2pbrt(path)

        # render mask image
        mask_path = os.path.join(MASK_PATH, get_fn(path))
        render_scene(learn_model_pbrt, mask_path, 256, 'mask')

    for path in model_paths:
        print(f"Rendering gbuffer for {path}")
        # convert obj to pbrt
        model_pbrt = obj2pbrt(path)

        # render position image
        position_path = os.path.join(POSITION_PATH, get_fn(path))
        render_scene(model_pbrt, position_path, 256, 'position')

        # render normal image
        normal_path = os.path.join(NORMAL_PATH, get_fn(path))
        render_scene(model_pbrt, normal_path, 256, 'normal')


if __name__ == '__main__':
    main()
