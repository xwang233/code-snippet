import os
import datetime
import apex
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

print(torch.__version__)

def run(rank, size):
    c = 2
    dtype = torch.float
    device = 'cuda'

    a = torch.randn(1, c, 3, 4, device='cuda:0', dtype=dtype) * 2 + 5

    # net = torch.nn.SyncBatchNorm(c)
    net = apex.parallel.SyncBatchNorm(c)

    net = net.cuda()
    ddp_net = DDP(net, device_ids=[0])

    b: torch.Tensor = ddp_net(a)

    # b.backward(torch.randn_like(b))

    torch.cuda.synchronize()

def init_processes(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    dist.init_process_group(backend, rank=rank, world_size=size, timeout=datetime.timedelta(seconds=10))
    try:
        fn(rank, size)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()

size = 2
processes = []

for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()