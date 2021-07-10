import torch
import itertools
import os
import sys
import shutil
import math
import time
import json

_ver = torch.__version__
ver = _ver[_ver.find('+')+1:]

data_path = f'data-{ver}'

if os.path.exists(data_path):
    shutil.rmtree(data_path)
os.mkdir(data_path)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = 'cuda'

DTYPES = [torch.float, torch.double]
MAX_DIMS = 6

idx = 0
d_perf = {}
NITER = 200

d_shapes = {}

for dtype in DTYPES:
    for tensor_dim in range(1, MAX_DIMS + 1):
        dims = list(range(tensor_dim))
        x0 = torch.randn(*([4]*tensor_dim), dtype=dtype, device=device)
        x1 = torch.randn(*([8]*tensor_dim), dtype=dtype, device=device)

        _perf_size = math.floor(math.pow(10_000_000, 1.0/tensor_dim))
        x_perf = torch.randn(*([_perf_size]*tensor_dim), dtype=dtype, device=device)

        for fft_dim_size in range(1, tensor_dim + 1):

            for fft_dim in itertools.combinations(dims, fft_dim_size):
                # print(tensor_dim, fft_dim_size, fft_dim)

                y = torch.fft.rfftn(x0, dim=fft_dim)
                torch.save(y, f'{data_path}/data-{idx}-0.pt')
                y = torch.fft.rfftn(x1, dim=fft_dim)
                torch.save(y, f'{data_path}/data-{idx}-1.pt')

                # warmup
                for _ in range(NITER): torch.fft.rfftn(x_perf, dim=fft_dim)

                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(NITER): torch.fft.rfftn(x_perf, dim=fft_dim)
                torch.cuda.synchronize()
                t_end = time.time()

                t_cost = (t_end - t_start) / NITER * 1e6
                print(idx, t_cost)

                d_perf[idx] = t_cost
                d_shapes[idx] = f'{dtype}, {tensor_dim = }, {fft_dim = }'

                idx += 1

torch.cuda.synchronize()

with open(f'perf-{ver}.json', 'w') as f:
    json.dump(d_perf, f, indent=2)

with open(f'shapes.json', 'w') as f:
    json.dump(d_shapes, f, indent=2)