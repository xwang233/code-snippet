import torch
from torch.testing import _compare_tensors_internal

import time
import random

def topKViaSort(x, k, dim):
    val, idx = x.sort(dim, True)
    return (val.narrow(dim, 0, k), idx.narrow(dim, 0, k))
def run(func, reps=100):
    # warmup
    for _ in range(reps):
        func()
    
    torch.cuda.synchronize()
    start_t = time.time()
    
    for _ in range(reps):
        y = func()
    
    torch.cuda.synchronize()
    end_t = time.time()
    
    return (end_t - start_t) / reps
    
def p(device, dtype, msg):
    print(torch.__version__)
    print('device:', device)
    
    ns = [1, 3, 10]
    bs = [10000, 50000, 100000, 500000, 1000000, 5000000, 1863557, 
          3298121, 1701861, 1171237, 2529899, 4629242, 1085346]
    ks = [1, 10, 50, 100, 500, 1000]
    
    sizes = set()
    for n1 in ns:
        for n2 in ns:
            for b in bs:
                for k in ks:
                    if (n1 == 10 and n2 == 10):
                        continue
                    sizes.add((n1, b,     k, -1))
                    sizes.add((n1, b, n2, k, 1))
                    sizes.add((b, n1,     k, 0))
    fn = f'{msg}-{device}-{dtype}.txt'
    print(fn)
    print(f'total {len(sizes)} sizes\n')
    
    t1 = time.time()
    count = 0

    with open(fn, 'w') as f:
        for size in sizes:
            torch.cuda.empty_cache()
            x = torch.randn(*size[:-2], device=device, dtype=dtype)

            t = run(lambda: torch.topk(x, k=size[-2], dim=size[-1]))
            t *= 1000   # time in ms
            f.write(f'{size} # {t}\n')

            y1 = torch.topk(x, k=size[-2], dim=size[-1])
            y2 = topKViaSort(x, k=size[-2], dim=size[-1])

            # values should be exactly equal
            a, b = _compare_tensors_internal(y1.values, y2[0], atol=0, rtol=0, equal_nan=False)
            assert a, b

            if not y1.indices.eq(y2[1]).all():
                vals = x.gather(index=y1.indices, dim=size[-1])
                a, b = _compare_tensors_internal(vals, y2[0], atol=0, rtol=0, equal_nan=False)
                assert a, b
            
            count += 1
            if count % (max(1, len(sizes) // 100)) == 0:
                print(f'{count} / {len(sizes)}, time cost = {time.time()-t1}, '
                      f'time left {(time.time()-t1)/count*(len(sizes)-count)}', flush=True)
                    
    t2 = time.time()
    print(f'total time {t2 - t1}')


# p('cuda', torch.float, 'before')

# p('cuda', torch.float, 'after')