import torch
import time
import itertools
import gc
import json
import multiprocessing
import threading
import math

from torch.testing._internal.common_utils import random_hermitian_pd_matrix
from torch.testing._core import _compare_tensors_internal

TIME_MULTIPLIER = 1e6
TIME_UNIT = 'us'

nb = 200
# nb = 1

device = 'cuda:0'
torch.cuda.set_device(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def compare(x, y, *, rtol, atol):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            raise RuntimeError("y tensor should be cuda, but it's not")
        return _compare_tensors_internal(x, y, rtol=rtol, atol=atol, equal_nan=False)
    
    a = True
    b = {}
    for x_, y_, s_ in zip(x, y, ['U', 'S', 'V']):
        a_, b_ = compare(x_, y_, rtol=rtol, atol=atol)

        a = a and a_
        if not a_:
            b[s_] = b_
    
    return a, json.dumps(b, indent=2)


def main(s: str = ''):
    def prof(b_, m_, n_, dtype=torch.float, p=None, key=None, out_file=None, out_js=None):
        gc.collect()
        torch.cuda.empty_cache()

        assert key is not None, "key can't be None"
        assert p is not None, "p can't be None"

        x = torch.randn(*b_, m_, n_, dtype=dtype, device=device).to(memory_format=torch.channels_last)
        # xc = x.clone().cpu()
        xc = x.clone().contiguous()

        torch.cuda.synchronize()

        # cpu timing
        t1 = time.time()
        for _ in range(nb):
            yc = p(xc)
        torch.cuda.synchronize()
        t2 = time.time()
        cpu_time = (t2-t1)/nb*TIME_MULTIPLIER
        # print('cpu', cpu_time, 'ms')

        if torch.isnan(yc).any():
            print('cpu output contains nan')

        # warmup
        for _ in range(nb):
            y_warmup = p(x)
        torch.cuda.synchronize()

        c, d = compare(xc, x, rtol=1e-7, atol=1e-7)
        if not c:
            print('original matrix compare')
            print(d)
            raise RuntimeError('original value x modified')

        torch.cuda.profiler.start()
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            y = p(x)
            torch.cuda.synchronize()
        torch.cuda.profiler.stop()

        torch.cuda.synchronize()

        # gpu timing
        t1 = time.time()
        for _ in range(nb):
            y = p(x)
        torch.cuda.synchronize()
        t2 = time.time()
        gpu_time = (t2-t1)/nb*TIME_MULTIPLIER
        # print('gpu', gpu_time, 'ms')


        # gpu timing 2
        t1 = time.time()
        for _ in range(nb):
            y_second = p(x.contiguous()).to(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        t2 = time.time()
        gpu_time_2 = (t2-t1)/nb*TIME_MULTIPLIER
        # print('gpu', gpu_time, 'ms')

        e, f = compare(y_warmup, y, rtol=0, atol=0)
        if not e:
            print('non-determinism: cholesky_solve value output')
            print(f)
            raise RuntimeError('non-deterministic output')

        a, b = compare(yc, y, rtol=1e-3, atol=1e-3)
        if not a:
            print('numerical mismatch: output value compare')
            print(b)

        print(f'{key}'.ljust(55) +
              f'{cpu_time :>10.3f} {gpu_time :>10.3f} {gpu_time_2 :>10.3f} {gpu_time_2 / gpu_time :>10.2f}')

        if out_file is not None:
            out_file.write(
              f'| {key}'
              f'| {cpu_time :>10.3f} | {gpu_time :>10.3f} | {gpu_time_2 :>10.3f} '
              f'| {gpu_time_2 / gpu_time :>10.2f} |\n')
        
        if out_js is not None:
            assert key not in out_js
            out_js[key] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'gpu_time_2': gpu_time_2,
            }

        torch.cuda.synchronize()
    
    print(s)
    print(torch.__version__)
    print()
    print('shapes'.ljust(40) +
         f'cont({TIME_UNIT}),   cl({TIME_UNIT}),   cl-cont-cl({TIME_UNIT}),   speed_up')

    name = 'before'

    out_js = {}
    with open(f'{name}.md', 'w') as out_file:
        out_file.write(
            f'| shapes | cont({TIME_UNIT}) | cl({TIME_UNIT}) | cl-cont-cl({TIME_UNIT}) | speed_up |\n')
        out_file.write('|' + '---|' * 5 + '\n')

        mode = 'bilinear'
        for b, n, mmul, omul in itertools.product(
            [[2**i1, 2**i2] for i1 in range(0, 2) for i2 in range(0, 7, 2)],
            [2**j for j in range(5, 9)],
            [0.5, 1, 2],
            [0.5, 1, 2]
        ):
            if b and b[0] * n >= 2**14:
                continue

            m = math.floor(n * mmul)

            om = math.floor(m * omul)
            on = math.floor(n * omul)

            prof(b, m, n,
                dtype=torch.half,
                p=lambda x: torch.nn.functional.interpolate(x, (om, on), mode=mode, align_corners=False),
                key=f'{(b, m, n, om, on)} half, AC False',
                out_file=out_file,
                out_js=out_js)
            prof(b, m, n,
                dtype=torch.float,
                p=lambda x: torch.nn.functional.interpolate(x, (om, on), mode=mode, align_corners=False),
                key=f'{(b, m, n, om, on)} float, AC False',
                out_file=out_file,
                out_js=out_js)
    
    with open(f'{name}.json', 'w') as fj:
        json.dump(out_js, fj, indent=2)

if __name__ == "__main__":
    main()

