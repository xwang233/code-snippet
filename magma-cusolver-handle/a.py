import torch
import time

print(torch.__version__)

niter = 500   # for benchmark timing
niter_prof = 5  # for nsys profiling

def prof(func, data, emit_nvtx=False, nvtx_msg=None):
    # warmup
    for _ in range(niter): func(data)

    # nsys profiling
    torch.cuda.profiler.start()
    if emit_nvtx:
        with torch.cuda.nvtx.range(nvtx_msg):
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                for _ in range(niter_prof): func(data)
    torch.cuda.profiler.stop()

    # profiling
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(niter): func(data)

    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / niter * 1000    # in ms


def main():
    p1 = torch.inverse

    magma_data = torch.randn(4, 32, 32, dtype=torch.float, device='cuda')
    cusolver_data = torch.randn(1, 32, 32, dtype=torch.float, device='cuda')

    time_magma_1 = prof(p1, magma_data, emit_nvtx=True, nvtx_msg='First magma')

    time_cusolver = prof(p1, cusolver_data, emit_nvtx=True, nvtx_msg='cusolver')

    time_magma_2 = prof(p1, magma_data, emit_nvtx=True, nvtx_msg='Second magma')

    print(f'{time_magma_1 = : .3f}')
    print(f'{time_magma_2 = : .3f}')



if __name__ == '__main__':
    main()

