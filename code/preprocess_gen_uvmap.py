import argparse
import math
import re
import os

def hemispherical_uv_map(inp_path, out_path, write_mtl):
    vertices_pool = []
    uv_pool = []

    with open(out_path, 'w+') as f:
        for og_line in open(inp_path, 'r'):
            line = og_line.strip()
            splitted = re.split('\ +', line)
            if splitted[0] == 'v':
                x,y,z = float(splitted[1]), float(splitted[2]), float(splitted[3])
                r = (x**2 + y**2 + z**2)**0.5
                u = math.atan2(y, z) / math.pi
                v = math.acos(x / r) / math.pi if r != 0. else 0.
                f.write(og_line)
                f.write('vt {} {}\n'.format(u, v))
            elif splitted[0] == 'vt':
                continue
            elif splitted[0] == 'vn':
                f.write(og_line)
            elif splitted[0] == 'f':
                fis = list()
                for i in range(1, len(splitted)):
                    vi,ui,ni = re.split('/', splitted[i])
                    fis.append('{}/{}/{}'.format(vi,vi,ni))
                f.write('f {}\n'.format(' '.join(fis)))
            else:
                if not write_mtl:
                    continue
                elif splitted[0] == 'map_Kd':
                    continue
                elif splitted[0] == 'map_Ka':
                    continue
                elif splitted[0] == 'map_Bump':
                    continue
                elif splitted[0] == 'usemtl' and splitted[1] == '(null)':
                    continue
                else:
                    f.write(og_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', type=str, required=True, help='Input mesh path')
    parser.add_argument('--out_path', type=str, required=True, help='Output mesh path')
    parser.add_argument('--write_mtl', action='store_true', help='Whether to write mtl info or no')
    opt = parser.parse_args()

    if not os.path.exists(opt.out_path):
        os.makedirs(opt.out_path)

    fn_list = os.listdir(opt.inp_path)
    for fn in fn_list:
        inp = os.path.join(opt.inp_path, fn)
        out = os.path.join(opt.out_path, fn)
        print("{} -> {}".format(inp, out))
        hemispherical_uv_map(inp, out, opt.write_mtl)
