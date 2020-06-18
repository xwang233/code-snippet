import sys
import gc
import traceback

import torch

n = 1
in_channels = 16
out_channels = 16
kernel_size = 3

sizes = [5 * int(1.1**i) for i in range(1000)]

opd = {
    '0': 'cudnn conv2d',
    '1': 'cuda max_pool2d',
    '2': 'torch.log',
    '3': 'torch.nn.ReLU'
}

def test(opi, set_to_none, clean_in_exception, req_grad):
    print('cuda OOM at size:')
    i = 0
    crash_count = 4

    while i < len(sizes):
        size = sizes[i]
        assert isinstance(size, int)

        if set_to_none:
            x = None
            op = None
            y = None
            if req_grad:
                g = None

        try:
            x = torch.randn(n, in_channels, size, size, dtype=torch.float, device='cuda', requires_grad=req_grad)

            if opi == 0:
                # cudnn
                op = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)\
                    .to(dtype=torch.float, device='cuda')
            elif opi == 1:
                # cuda kernel without cudnn, may have internal tensor
                op = torch.nn.MaxPool2d(kernel_size=kernel_size)
            elif opi == 2:
                # aten native, no weight
                op = torch.log
            elif opi == 3:
                # nn, no weight
                op = torch.nn.ReLU()
            else:
                raise RuntimeError('unknown opi')

            y = op(x)
            if req_grad:
                g = torch.randn_like(y)
                y.backward(g)
        except RuntimeError as e:
            if not str(e).startswith('CUDA out of memory'):
                raise
            # RuntimeError: CUDA out of memory

            torch.cuda.synchronize()
            if clean_in_exception:
                del x
                del op
                del y
                if req_grad:
                    del g
                gc.collect()
                torch.cuda.empty_cache()

            print(size)
            # traceback.print_exc()

            crash_count -= 1
            if crash_count == 0:
                break

            i = 0
            continue

        i += 1

if __name__ == "__main__":
    assert len(sys.argv) == 5

    print(f'op {opd[sys.argv[1]]}')
    print(f'set_to_none {sys.argv[2]}')
    print(f'clean_in_exception {sys.argv[3]}')
    print(f'req_grad {sys.argv[4]}')

    test(
        int(sys.argv[1]), 
        sys.argv[2] == 'True', 
        sys.argv[3] == 'True', 
        sys.argv[4] == 'True'
    )

    print()