import torch
import torch.distributed as dist
import random
import gc
import time
import traceback
import argparse
import sys
from collections import Counter
import json
import os

from torch.testing._core import _compare_tensors_internal

def main(rank: int, num_random_shapes: int):
    if rank == 0:
        print(torch.__version__)
        print(f'cudnn_ver: {torch.backends.cudnn.version()}')

    report_every = min(1000, max(1, num_random_shapes // 20))
    t_start = time.time()

    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    print(f'#{rank}', torch.cuda.get_device_name(rank))

    calculated_shapes = 0
    mismatches = [0, 0, 0] # forward, dgrad, wgrad
    exception_shapes = 0
    con_exceptions = Counter()

    d_size = {
        torch.float: 4,
        torch.half: 2
    }

    while calculated_shapes < num_random_shapes:
        n = random.randint(1, 8)
        c = random.randint(8 // 8, 512 // 8) * 8
        d = random.randint(8 // 4, 512 // 4) * 4
        # h = random.randint(4, 512)
        # w = random.randint(4, 512)
        h = d
        w = d

        if random.random() > 0.5:
            c = random.choice([4, 8, 16, 32, 64, 128, 256])
            oc = c
            g = c
        else:
            oc = random.randint(4, 1024)
            g = 1

        ks = random.choice([1, 2, 3, 5])
        dtype = random.choice([torch.half, torch.float])

        tensor_size = (
            2 * 2 * n * c * d * h * w + # (input + input.grad) and ref
            2 * 2 * oc * c * ks **3 +  # (weight + weight.grad) and ref
            2 * 2 * n * oc * d * h * w # (output (est) + output.grad) and ref
            ) * d_size[dtype] / 1e9
        if tensor_size > 4.0:
            # print('tensor too large, continue')
            continue

        oom = False

        try:
            x = torch.randn(n, c, d, h, w, dtype=dtype, device=device).requires_grad_()
            x.to(memory_format=torch.channels_last_3d)
            ref_cont_x = x.detach().clone().contiguous().requires_grad_()

            conv = torch.nn.Conv3d(c, oc, kernel_size=ks, stride=1, padding=0, dilation=1, groups=g)
            conv = conv.to(dtype=dtype, device=device, memory_format=torch.channels_last_3d)
            ref_cont_conv = torch.nn.Conv3d(c, oc, kernel_size=ks, stride=1, padding=0, dilation=1, groups=g)
            ref_cont_conv = ref_cont_conv.to(dtype=dtype, device=device, memory_format=torch.channels_last_3d)

            with torch.no_grad():
                for p, rp in zip(conv.parameters(), ref_cont_conv.parameters()):
                    rp.copy_(p)

            out = conv(x)
            ref_cont_out = ref_cont_conv(ref_cont_x)

            out.sum().backward()
            ref_cont_out.sum().backward()

            _a, _b = _compare_tensors_internal(out, ref_cont_out, atol=1e-3, rtol=1e-3, equal_nan=False)
            if not _a:
                mismatches[0] += 1
            _c, _d = _compare_tensors_internal(x.grad, ref_cont_x.grad, atol=1e-3, rtol=1e-3, equal_nan=False)
            if not _c:
                mismatches[1] += 1
            _e, _f = _compare_tensors_internal(conv.weight.grad, ref_cont_conv.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=False)
            if not _e:
                mismatches[2] += 1
        except RuntimeError as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if str(e).startswith('CUDA out of memory'):
                oom = True
                # print(f'******************OOM {n=} {c=} {d=} {h=} {w=} {oc=} {g=} {ks=} {dtype=}')
            else:
                print(f'*************** {rank=} {n=} {c=} {d=} {h=} {w=} {oc=} {g=} {ks=}\n'
                      f'*************** {dtype=} {tensor_size=:.3f} GB\n'
                      f'*************** {exc_type}: {exc_value}')
                con_exceptions[str(exc_value)] += 1

                # traceback.print_exc()
                exception_shapes += 1
                # raise

        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue

        calculated_shapes += 1

        if calculated_shapes // report_every != (calculated_shapes - 1) // report_every:
            t_now = time.time()
            tl_est = (num_random_shapes - calculated_shapes) * (t_now - t_start) / calculated_shapes
            print(f'#{rank} time cost = {t_now - t_start :.3f}, {calculated_shapes = }, time left (est) = {tl_est :.3f}')

    print(f'#{rank} Exceptions:', json.dumps(con_exceptions, indent=2))

    dist.barrier()

    l_report = [*mismatches, exception_shapes]
    t_report = torch.Tensor(l_report).cuda()
    dist.reduce(t_report, dst=0)
    if rank == 0:
        print()
        print(f'total shapes = {num_random_shapes * dist.get_world_size()}')
        print('mismatches: forward, dgrad, wgrad; num of exception shapes')
        print(t_report.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--total-shapes', type=int, default=10)
    args = parser.parse_args()

    print(args)

    dist.init_process_group(backend='nccl', init_method='env://')

    main(args.local_rank, args.total_shapes // dist.get_world_size())
