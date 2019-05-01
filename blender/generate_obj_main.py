import argparse
import os
import subprocess

CAR_DIR='/data/graphics/toyota-pytorch/toyota_files_1029_untarred/afs/csail.mit.edu/u/h/helenwh/toyota_files/'
BLENDER='/home/budmonde/opt/blender/blender'

parser = argparse.ArgumentParser()
parser.add_argument('--models_fn', type=str, default='./car_models.txt', help='car model fnames')
parser.add_argument('--output_dir_full', type=str, default='./output_full/', help='output directory name of full meshes')
parser.add_argument('--output_dir_learn', type=str, default='./output_learn/', help='output directory name of learnable meshes')
opt = parser.parse_args()

if not os.path.exists(opt.output_dir_full):
    os.makedirs(opt.output_dir_full)
if not os.path.exists(opt.output_dir_learn):
    os.makedirs(opt.output_dir_learn)

with open(opt.models_fn) as f:
    car_list = [line.strip().split(' ') for line in f.readlines()]

for car in car_list:
    # Format: (<relative_path>/<filename>.blend, <tex_label>)
    absolute_path = os.path.join(CAR_DIR, car[0])
    filename = absolute_path.split('/')[-1].split('.')[0]
    tex_label = car[1]
    subprocess.run([
        BLENDER,
            '--background', absolute_path,
            '--python', 'generate_obj_blender.py',
            '--',
                '--output_dir_full', opt.output_dir_full,
                '--output_dir_learn', opt.output_dir_learn,
                '--filename', filename,
                '--tex_label', tex_label,
                #'--decrease_vertices',
                '--write_mtl'],
        check=True)
