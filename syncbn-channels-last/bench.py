from typing import List

import subprocess
import itertools
import shutil
import os
import json
import time
import argparse
import torch

import utils

NPROC_PER_NODE = 2


def silent_remove(f):
    try:
        os.remove(f)
    except OSError:
        pass


def get_shapes():
    shapes = set()
    limit = [2**12 * 16, 2 * 1024**3]
    for n, c, hw in itertools.product(
        [2**i for i in range(2, 9)],
        [2**j for j in range(2, 12)],
        [16, 32, 56, 112, 224]
    ):
        if limit[0] <= n*c * 16 <= limit[1]:
            shapes.add((n, c))
        if limit[0] <= n*c*hw*hw * 16 < limit[1]:
            shapes.add((n, c, hw, hw))

    return shapes


# nanoseconds
def get_kernel_average_time(ses: 'List[str]', s: str):
    for line in ses:
        if s in line:
            times = line.split()
            avg = float(times[3])

            return avg
    return -1


def main(j, pre: bool):
    shapes = get_shapes()

    python_exe = shutil.which('python3'); assert python_exe is not None
    nsys_exe = shutil.which('nsys'); assert nsys_exe is not None

    kw = utils.kw

    total = len(shapes)
    done = 0
    time_start = time.time()

    for shape in shapes:
        shape_str = [str(s) for s in shape]
        ss = str(shape)
        j[ss] = {}

        p = subprocess.run([
            nsys_exe,
            'nvprof',
            '--profile-from-start', 'off',
            '-f',
            '-o', 'report',
            python_exe,
            '-m', 'torch.distributed.launch', f'--nproc_per_node={NPROC_PER_NODE}',
            'sbn-kernel.py',
            '--shape', *shape_str,
            '--pre', str(pre)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if int(p.returncode) != 0:
            print('#'*25, 'so\n', p.stdout.decode('ascii'))
            print('#'*25, 'se\n', p.stderr.decode('ascii'))
            p.check_returncode()

        ses = p.stderr.decode('ascii').split('\n')

        for torch_func in kw:
            j[ss][torch_func] = {}
            for kernel in kw[torch_func]:
                t = get_kernel_average_time(ses, kernel)
                # print(torch_func + ' -- ' + kernel, t)

                j[ss][torch_func][kernel] = t

        done += 1
        t_cost = time.time() - time_start
        print(f'{done} / {total} done,',
              f'time cost = {t_cost : .2f} s',
              f'time left = {t_cost / done * (total - done) : .2f} s')
        print(ss, json.dumps(j[ss], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', action='store_true', default=False)
    args = parser.parse_args()
    
    pre = 1 if args.pre else 0

    print('pytorch version', torch.__version__)
    j = {}

    try:
        main(j, pre)
    except:
        raise
    finally:
        silent_remove('report.qdrep')
        silent_remove('report.sqlite')
        with open(f'output-{torch.__version__}.json', 'w') as f:
            json.dump(j, f, indent=2, sort_keys=True)
