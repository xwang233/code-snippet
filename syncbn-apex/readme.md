## sync batchnorm apex

tested on pytorch master

- before https://github.com/pytorch/pytorch/commit/8066fba2260fe46949680294b78b372067233746
  - \[RELAND2\] Change AccumulateGrad to yield `.grad`s that match weights' memory layout #40358
  - good
- before https://github.com/pytorch/pytorch/commit/b05c34259b5e3ff9b55fba5a24d7c5f2f2036471
  - relax size check in flatten_for_scatter_gather (#40573)
  - crash
```
1.6.0a0+e180ca6
Process Process-2:
Traceback (most recent call last):
  File "/usr/lib64/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/usr/lib64/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "sbn.py", line 36, in init_processes
    fn(rank, size)
  File "sbn.py", line 24, in run
    b: torch.Tensor = ddp_net(a)
  File "/home/xwang/Developer/pytorch/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xwang/Developer/pytorch/torch/nn/parallel/distributed.py", line 507, in forward
    output = self.module(*inputs[0], **kwargs[0])
  File "/home/xwang/Developer/pytorch/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xwang/.local/lib/python3.7/site-packages/apex/parallel/optimized_sync_batchnorm.py", line 85, in forward
    return SyncBatchnormFunction.apply(input, z, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.training or not self.track_running_stats, exponential_average_factor, self.process_group, channel_last, self.fuse_relu)
  File "/home/xwang/.local/lib/python3.7/site-packages/apex/parallel/optimized_sync_batchnorm_kernel.py", line 36, in forward
    torch.distributed.all_gather(mean_l, mean, process_group)
  File "/home/xwang/Developer/pytorch/torch/distributed/distributed_c10d.py", line 1185, in all_gather
    work = _default_pg.allgather([tensor_list], [tensor])
RuntimeError: All tensor operands to scatter/gather must have the same size
Process Process-1:
Traceback (most recent call last):
  File "/usr/lib64/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/usr/lib64/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "sbn.py", line 36, in init_processes
    fn(rank, size)
  File "sbn.py", line 24, in run
    b: torch.Tensor = ddp_net(a)
  File "/home/xwang/Developer/pytorch/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xwang/Developer/pytorch/torch/nn/parallel/distributed.py", line 507, in forward
    output = self.module(*inputs[0], **kwargs[0])
  File "/home/xwang/Developer/pytorch/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xwang/.local/lib/python3.7/site-packages/apex/parallel/optimized_sync_batchnorm.py", line 85, in forward
    return SyncBatchnormFunction.apply(input, z, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.training or not self.track_running_stats, exponential_average_factor, self.process_group, channel_last, self.fuse_relu)
  File "/home/xwang/.local/lib/python3.7/site-packages/apex/parallel/optimized_sync_batchnorm_kernel.py", line 36, in forward
    torch.distributed.all_gather(mean_l, mean, process_group)
  File "/home/xwang/Developer/pytorch/torch/distributed/distributed_c10d.py", line 1185, in all_gather
    work = _default_pg.allgather([tensor_list], [tensor])
RuntimeError: All tensor operands to scatter/gather must have the same size
```
- after https://github.com/pytorch/pytorch/commit/b05c34259b5e3ff9b55fba5a24d7c5f2f2036471
  - relax size check in flatten_for_scatter_gather (#40573)
  - good

## use torch.nn sync batchnorm

```python
    # net = torch.nn.SyncBatchNorm(c)
    net = apex.parallel.SyncBatchNorm(c)
```

no crash