import torch
import time
import itertools
import gc
import json

from torch.testing._core import _compare_tensors_internal

# nb = 100
nb = 1

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


def main(s: str):
    def prof(b_, n_, dtype=torch.float, p=None, flag=None, is_linalg=False):
        # gc.collect()
        # torch.cuda.empty_cache()
        if p is None:
            p = lambda x: x

        # print(b_, n_)

        def _prof(x: torch.Tensor):
            xc = x.clone().cpu()

            # cpu timing
            t1 = time.time()
            for _ in range(nb):
                yc = p(xc)
            t2 = time.time()
            cpu_time = (t2-t1)/nb*1e3
            # print('cpu', cpu_time, 'ms')

            # warmup
            for _ in range(nb):
                y_warmup = p(x)
            torch.cuda.synchronize()

            # torch.cuda.nvtx.range_push('checking if async')
            # big = torch.randn(1, 1000, 1000, 1000, dtype=torch.half, device='cuda')
            # torch.exp(big)
            # y = p(x)
            # torch.exp(big)
            # torch.cuda.nvtx.range_pop()

            c, d = compare(xc, x, rtol=1e-7, atol=1e-7)
            if not c:
                print('original matrix compare')
                print(d)
                raise RuntimeError('original value modified')
        
            # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as profx:
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                y = p(x)
                torch.cuda.synchronize()
            # if b_[0] == 4:
            #     print(profx.table())
            #     profx.export_chrome_trace('./chrome.ctr')

            torch.cuda.synchronize()

            # gpu timing
            t1 = time.time()
            for _ in range(nb):
                # y = torch.svd(x)
                y = p(x)
            torch.cuda.synchronize()
            t2 = time.time()
            gpu_time = (t2-t1)/nb*1e3
            # print('gpu', gpu_time, 'ms')

            e, f = compare(y_warmup, y, rtol=0, atol=0)
            if not e:
                print('non-determinism: svd value output')
                print(f)
                raise RuntimeError('non-deterministic output')

            # Note that the reconstruct is `A = U @ S @ V^H`
            if is_linalg:
                # torch.linalg.svd returns U, S, V^H
                reconstruct = torch.matmul(y[0],
                                           torch.diag_embed(y[1]).to(dtype=dtype))\
                                    .matmul(y[2])
            else:
                # torch.svd returns U, S, V
                reconstruct = torch.matmul(y.U,
                                           torch.diag_embed(y.S).to(dtype=dtype))\
                                    .matmul(y.V.conj().transpose(-1, -2))

            a, b = compare(x, reconstruct, rtol=1e-3, atol=1e-3)
            # a, b = compare(yc, y, rtol=1e-3, atol=1e-3)
            if not a:
                print('numerical mismatch: svd value compare')
                print(b)
                # raise RuntimeError()

            print(f'{b_} {x.size(-2)} {x.size(-1)} {dtype}'.ljust(35) + f'{cpu_time : .3f}  {gpu_time : .3f}' + f'     {is_linalg}')
            # f.write(f'{b_} {n_} {dtype}; ' + f'{cpu_time : .3e}, {gpu_time : .3e}\n')
            torch.cuda.synchronize()
        
        _prof(torch.randn(*b_, n_, n_, device='cuda', dtype=dtype))
        _prof(torch.randn(*b_, 2*n_, n_, device='cuda', dtype=dtype))
        _prof(torch.randn(*b_, n_, 2*n_, device='cuda', dtype=dtype))
    
    print(s)
    print(torch.__version__)
    print()
    print('batch_size, matrix_size, dtype'.ljust(35) + 'cpu_time(ms), gpu_time(ms), is_linalg')

    # p1 = torch.svd
    # for dtype in [torch.float, torch.double]:
    #     prof([4], 256, p=p1, flag=0, dtype=dtype)
    #     prof([], 256, p=p1, dtype=dtype)
    #     prof([4], 256, p=p1, flag=1, dtype=dtype)
    #     prof([2], 64, p=p1, dtype=dtype)
    #     prof([2], 128, p=p1, dtype=dtype)
    #     prof([2], 256, p=p1, dtype=dtype)
    #     prof([2], 512, p=p1, dtype=dtype)
    #     prof([2], 1024, p=p1, dtype=dtype)

    for b, n in itertools.product(
        [[]] + [[2**i] for i in range(11)],
        [2**j for j in range(1, 11, 1)]
    ):
        if b and b[0] * n >= 2**12:
            continue
        prof(b, n, p=lambda x: torch.linalg.svd(x, full_matrices=False), is_linalg=True)
        prof(b, n, p=torch.svd, is_linalg=False)

    # prof([], 1536, p=torch.svd)
    # prof([], 2048, p=torch.svd)
    # prof([], 4096, p=torch.svd)

    # b = [2, 3]
    # n = 256
    # data = torch.randint(1, 10, (*b, n, n), dtype=torch.float64, device='cuda')
    # with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=True):
    #     inv_ref = torch.svd(data)

    # for i in range(100):
    #     inv = torch.svd(data)
    #     a, b = torch.testing._compare_tensors_internal(inv_ref, inv, atol=0, rtol=0, equal_nan=False)
    #     assert a, f'i = {i}, {b}'
    
    # print('good, no non-determinism')

if __name__ == "__main__":
    main('after')

