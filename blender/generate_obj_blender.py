import sys
import argparse
import os
from math import radians

import bpy
import bmesh

# Parse Args
argv = sys.argv
if '--' not in argv:
    argv = []
else:
    argv = argv[argv.index('--') + 1:] # get all args after '--'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir_full', required=True, help='output dir for full mesh')
parser.add_argument('--output_dir_learn', required=True, help='output dir for learnable mesh')
parser.add_argument('--filename', required=True, help='filename')
parser.add_argument('--tex_label', required=True, help='learneable tex label')
parser.add_argument('--decrease_vertices', action='store_true', help='whether to decrease vertex count or not')
parser.add_argument('--write_mtl', action='store_true', help='whether to write materials or not')
opt = parser.parse_args(argv)

# Decrease vertex count
if opt.decrease_vertices:
    ctx = bpy.context

    meshes = [o.data for o in ctx.selected_objects if o.type == 'MESH']

    bm = bmesh.new()

    for m in meshes:
        bm.from_mesh(m)
        bmesh.ops.dissolve_limit(
            bm, angle_limit=radians(1.7), verts=bm.verts, edges=bm.edges)
        bm.to_mesh(m)
        m.update()
        bm.clear()

    bm.free()

# Export Scene
filepath = os.path.join(opt.output_dir_full, '{}.obj'.format(opt.filename))
bpy.ops.export_scene.obj(
    filepath=filepath,
    check_existing=True,
    axis_forward='-Z',
    axis_up='Y',
    filter_glob='*.obj;*.mtl',
    use_selection=False,
    use_animation=False,
    use_mesh_modifiers=True,
    use_edges=False,
    use_smooth_groups=False,
    use_smooth_groups_bitflags=False,
    use_normals=True,
    use_uvs=False,
    use_materials=opt.write_mtl,
    use_triangles=True,
    use_nurbs=False,
    use_vertex_groups=False,
    use_blen_objects=False,
    group_by_object=False,
    group_by_material=False,
    keep_vertex_order=False,
    global_scale=1.0,
    path_mode='AUTO'
)

# Remove unlearneable labels
if opt.tex_label != '':
    ref_mtl = bpy.data.materials[opt.tex_label]
    for obj in bpy.data.objects:
        select = True
        for slot in obj.material_slots:
            select = select and slot.material != ref_mtl
        obj.select = select
    bpy.ops.object.delete()

# Export Scene
filepath = os.path.join(opt.output_dir_learn, '{}.obj'.format(opt.filename))
bpy.ops.export_scene.obj(
    filepath=filepath,
    check_existing=True,
    axis_forward='-Z',
    axis_up='Y',
    filter_glob='*.obj;*.mtl',
    use_selection=False,
    use_animation=False,
    use_mesh_modifiers=True,
    use_edges=False,
    use_smooth_groups=False,
    use_smooth_groups_bitflags=False,
    use_normals=True,
    use_uvs=False,
    use_materials=opt.write_mtl,
    use_triangles=True,
    use_nurbs=False,
    use_vertex_groups=False,
    use_blen_objects=False,
    group_by_object=False,
    group_by_material=False,
    keep_vertex_order=False,
    global_scale=1.0,
    path_mode='AUTO'
)
