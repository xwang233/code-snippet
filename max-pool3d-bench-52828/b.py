import torch
import torch.utils.benchmark as benchmark
import pickle

fuzzer = benchmark.Fuzzer(parameters=[
    benchmark.FuzzedParameter('n', minval=4, maxval=16, distribution='uniform'),
    benchmark.FuzzedParameter('c', minval=4, maxval=256, distribution='uniform'),
    benchmark.FuzzedParameter('d', minval=8, maxval=256, distribution='uniform'),
    benchmark.FuzzedParameter('h', minval=8, maxval=256, distribution='uniform'),
    benchmark.FuzzedParameter('w', minval=8, maxval=256, distribution='uniform'),
],
                          tensors=[
                              benchmark.FuzzedTensor('x',
                                                     size='ncdhw',
                                                     min_elements=12,
                                                     max_elements=10000000,
                                                     cuda=True,
                                                     dtype=torch.half,
                                                     max_allocation_bytes=1_000_000_000)
                          ],
                          seed=42)

res = []

for kernel_size in [2, 3, 5]:
    for tensors, tensor_params, params in fuzzer.take(20):
        sub_label = str(tensors['x'].size())
        res.append(
            benchmark.Timer(stmt=f'torch.nn.functional.max_pool3d(x, {kernel_size})',
                            setup='',
                            globals=tensors,
                            label=f'max_pool3d, {kernel_size=}',
                            sub_label=sub_label,
                            description=f'{torch.__version__}').blocked_autorange(min_run_time=0.1))

torch_ver = str(torch.__version__)
torch_git_ver = torch_ver[torch_ver.index('+') + 1:]

with open(f'{torch_git_ver}.pkl', 'wb') as f:
    pickle.dump(res, f)

compare = benchmark.Compare(res)
# compare.colorize()
compare.print()
