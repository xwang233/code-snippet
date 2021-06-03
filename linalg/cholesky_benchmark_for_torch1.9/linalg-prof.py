import torch
import time
import itertools
import gc
import json

TIME_MULTIPLIER = 1e6
TIME_UNIT = 'us'

nb = 200

torch.manual_seed(42)
torch.cuda.manual_seed(42)

_torch_ver = torch.__version__
torch_ver = _torch_ver[_torch_ver.index('+')+1:]
print(torch_ver)

j = {}

def main(s: str = ''):
    def prof(b_, n_, dtype=torch.float, p=None, flag=None):
        gc.collect()
        torch.cuda.empty_cache()

        _x = torch.randn(*b_, n_, n_, device='cuda', dtype=dtype)
        x = torch.matmul(_x, _x.transpose(-2, -1)) \
            + torch.eye(n_, n_, dtype=dtype, device='cuda') * 1e-3

        # warmup
        for _ in range(nb):
            y_warmup = p(x)
        torch.cuda.synchronize()

        # gpu timing
        t1 = time.time()
        for _ in range(nb):
            y = p(x)
        torch.cuda.synchronize()
        t2 = time.time()
        gpu_time = (t2 - t1) / nb * TIME_MULTIPLIER

        print(b_, n_, dtype, gpu_time, sep='\t')

        if len(b_) == 0:
            b_ = [0]
        
        key = str([b_[0], n_])
        value = gpu_time

        assert key not in j
        j[key] = value
        

    print(s)
    print(torch.__version__)
    print()
    print('batch_size, matrix_size, dtype'.ljust(35) + f'gpu_time({TIME_UNIT})')


    for n in range(2, 12):
        prof([], 2**n, p=torch.linalg.cholesky)
    
    for b in range(0, 11):
        prof([2**b], 4, p=torch.linalg.cholesky)
        prof([2**b], 32, p=torch.linalg.cholesky)
        prof([2**b], 128, p=torch.linalg.cholesky)
    
    # print(json.dumps(j, indent=2))
    with open(f'res-{torch_ver}.json', 'w') as f:
        json.dump(j, f, indent=2)

if __name__ == "__main__":
    main()
