import argparse
import os
import sys

import bpy
import math

def main():
    # Parse Args
    argv = sys.argv
    if '--' not in argv:
        argv = []
    else:
        argv = argv[argv.index('--') + 1:] # get all args after '--'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='input path')
    parser.add_argument('--output_path', required=True, help='output path')
    opt = parser.parse_args(argv)

    # Init empty blender scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = 'CYCLES'

    # Import obj
    bpy.ops.import_scene.obj(filepath=opt.input_path)

    # Load crevice dirt shader
    with bpy.data.libraries.load('./dirty_paint.blend', link=True) as (data_src, data_dst):
        data_dst.node_groups = ['crevice_dirt']

    # TODO: Not sure what this does. should try disabling later
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and len(obj.data.vertices) > 0:
            bpy.context.scene.objects.active = obj
            try:
                # The clean_angle affects the level of dirt a vertex gets
                bpy.ops.paint.vertex_color_dirt(clean_angle=math.pi/2.)
            except RuntimeError:
                continue

    # Next, combine the dirt with the given material
    material = bpy.data.materials.get('car_paint')

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Set diffuse node to white
    for node in nodes:
        if node.name.startswith('Mix.002'):
            node.inputs[1].default_value = [0.8, 0.8, 0.8, 1.0]


    # Find the output node
    for node in nodes:
        if node.name.startswith('Material Output'):
            output = node

    # Create crevice dirt node
    crevice_dirt = nodes.new(type='ShaderNodeGroup')
    crevice_dirt.node_tree = bpy.data.node_groups['crevice_dirt']
    # Set up the dirt parameters
    crevice_dirt.inputs[ 0].default_value = 0.8
    crevice_dirt.inputs[ 1].default_value = 1.0
    crevice_dirt.inputs[ 2].default_value = 0.8
    crevice_dirt.inputs[ 3].default_value = 1.0
    crevice_dirt.inputs[ 4].default_value = 0.8
    crevice_dirt.inputs[ 5].default_value = 1.0
    crevice_dirt.inputs[ 6].default_value = 0.4
    crevice_dirt.inputs[ 7].default_value = 0.8
    crevice_dirt.inputs[ 8].default_value = 0.0
    crevice_dirt.inputs[ 9].default_value = 0.2
    crevice_dirt.inputs[10].default_value = 0.1

    # Create mixer shader
    mix = nodes.new(type='ShaderNodeMixShader')

    # Find the current output shader
    current_output = None
    for l in links:
        if l.to_node.name.startswith('Material Output'):
            if l.to_socket == l.to_node.inputs[0]:
                current_output = l.from_socket
                break

    # Re-wire the nodes
    links.new(crevice_dirt.outputs[0], mix.inputs[2])
    links.new(crevice_dirt.outputs[1], mix.inputs[0])
    links.new(current_output, mix.inputs[1])
    links.new(mix.outputs[0], output.inputs[0])

    # Bake the texture onto an image
    bpy.ops.image.new(name='dirtmap')
    img = bpy.data.images['dirtmap']

    im_node = nodes.new(type='ShaderNodeTexImage')
    im_node.image = img
    im_node.select = True
    material.node_tree.nodes.active = im_node

    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

    # Save the image
    fn = opt.input_path.split('/')[-1].split('.')[0]
    img.filepath_raw = os.path.join(opt.output_path, '{}.png'.format(fn))
    img.file_format = 'PNG'
    img.save()

if __name__ == '__main__':
    main()
