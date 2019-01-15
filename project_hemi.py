import math
import re

def project_hemi(inp_fn, out_fn):
    vertices_pool = []
    uv_pool = []

    with open(out_fn, 'w+') as f:
        for og_line in open(inp_fn, 'r'):
            line = og_line.strip()
            splitted = re.split('\ +', line)
            if splitted[0] == 'v':
                x,y,z = float(splitted[1]), float(splitted[2]), float(splitted[3])
                r = (x**2 + y**2 + z**2)**0.5
                u = math.atan2(y, z) / math.pi
                v = math.acos(x / r) / math.pi
                f.write(og_line)
                f.write('vt {} {}\n'.format(u, v))
            elif splitted[0] == 'vt':
                continue
            elif splitted[0] == 'f':
                fis = list()
                for i in range(1, len(splitted)):
                    vi,ui,ni = re.split('/', splitted[i])
                    fis.append('{}/{}/{}'.format(vi,vi,ni))
                f.write('f {}\n'.format(' '.join(fis)))
            else:
                f.write(og_line)

project_hemi('datasets/scenes/octavia_clean.obj', 'datasets/scenes/uv_debug.obj')
