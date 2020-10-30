import apex
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

niter = 100


def run(rank, shape, dtype, pre):
    device = f'cuda:{rank}'
    c = shape[1]

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    def profile(net, mf, s: str):
        is_apex = 'apex' in s

        a = torch.randn(*shape, device=device, dtype=dtype).requires_grad_()
        if mf == torch.channels_last and len(shape) == 4:
            if is_apex:
                a = a.permute(0, 2, 3, 1)
            else:
                a = a.to(memory_format=torch.channels_last)

        net_ddp = DDP(net.cuda(rank), device_ids=[rank], output_device=rank)
        out = net_ddp(a)
        g = torch.randn_like(out)

        for _ in range(niter):
            net_ddp(a)
            out.backward(g, retain_graph=True)
        torch.cuda.synchronize()

    _a = torch.randn(*shape, device=device, dtype=dtype)
    _net_torch = torch.nn.SyncBatchNorm(c)
    _net_apex_contiguous = apex.parallel.SyncBatchNorm(c)
    _net_apex_channels_last = apex.parallel.SyncBatchNorm(c, channel_last=True)

    # warm up
    net_warmup = DDP(torch.nn.SyncBatchNorm(c).cuda(rank), device_ids=[rank],
                     output_device=rank)
    for _ in range(niter):
        net_warmup(_a)

    torch.cuda.profiler.start()
    if pre:
        profile(_net_torch, torch.contiguous_format, 'torch native contiguous')
    else:
        profile(_net_torch, torch.channels_last, 'torch native channels_last')
        # profile(_net_apex_contiguous, torch.contiguous_format, 'apex contiguous')
        profile(_net_apex_channels_last, torch.channels_last, 'apex channels_last')
    torch.cuda.profiler.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--shape', type=int, nargs='*')
    parser.add_argument('--pre', type=int)
    args = parser.parse_args()

    rank = args.local_rank
    shape = args.shape
    dtype = torch.float
    pre = args.pre

    print(args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)

    run(rank, shape, dtype, pre)


if __name__ == "__main__":
    main()
