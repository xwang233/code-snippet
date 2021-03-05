import torch
import time
import itertools
import gc
import json

from torch.testing._internal.common_utils import random_symmetric_pd_matrix

TIME_MULTIPLIER = 1e6
TIME_UNIT = 'us'

nb = 200
# nb = 1

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def compare(x, y, *, rtol, atol):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            raise RuntimeError("y tensor should be cuda, but it's not")
        return torch.testing._compare_tensors_internal(x, y, rtol=rtol, atol=atol, equal_nan=False)
    
    a = True
    b = {}
    for x_, y_, s_ in zip(x, y, ['U', 'S', 'V']):
        a_, b_ = compare(x_, y_, rtol=rtol, atol=atol)

        a = a and a_
        if not a_:
            b[s_] = b_
    
    return a, json.dumps(b, indent=2)


def main(s: str = ''):
    def prof(b_, n_, dtype=torch.float, p=None, flag=None):
        gc.collect()
        torch.cuda.empty_cache()

        if p is None:
            p = lambda x: x

        # print(b_, n_)
        # x = torch.randn(*b_, n_, n_, device='cuda', dtype=dtype)
        x = random_symmetric_pd_matrix(n_, *b_, device='cuda').to(dtype=dtype)

        xc = x.clone().cpu()

        # cpu timing
        t1 = time.time()
        for _ in range(nb):
            yc = p(xc)
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
            raise RuntimeError('original value modified')

        torch.cuda.profiler.start()
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            y = p(x)
            torch.cuda.synchronize()
        torch.cuda.profiler.stop()

        torch.cuda.synchronize()

        # gpu timing
        t1 = time.time()
        for _ in range(nb):
            # y = torch.cholesky(x)
            y = p(x)
        torch.cuda.synchronize()
        t2 = time.time()
        gpu_time = (t2-t1)/nb*TIME_MULTIPLIER
        # print('gpu', gpu_time, 'ms')

        e, f = compare(y_warmup, y, rtol=0, atol=0)
        if not e:
            print('non-determinism: cholesky value output')
            print(f)
            raise RuntimeError('non-deterministic output')

        reconstruct = torch.matmul(y, y.transpose(-1, -2))
        a, b = compare(x, reconstruct, rtol=1e-3, atol=1e-3)
        # a, b = compare(yc, y, rtol=1e-3, atol=1e-3)
        if not a:
            print('numerical mismatch: cholesky value compare')
            print(b)

        print(f'{b_} {n_} {dtype}'.ljust(35) + f'{cpu_time : .3f}  {gpu_time : .3f}')
        # f.write(f'{b_} {n_} {dtype}; ' + f'{cpu_time : .3e}, {gpu_time : .3e}\n')
        torch.cuda.synchronize()
    
    print(s)
    print(torch.__version__)
    print()
    print('batch_size, matrix_size, dtype'.ljust(35) +
         f'cpu_time({TIME_UNIT}), gpu_time({TIME_UNIT})')

    for b, n in itertools.product(
        [[]] + [[2**i] for i in range(11)],
        [2**j for j in range(1, 12, 1)]
    ):
        if b and b[0] * n >= 2**14:
            continue
        prof(b, n, p=torch.cholesky)

if __name__ == "__main__":
    main()

