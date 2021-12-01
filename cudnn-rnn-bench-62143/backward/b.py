import torch
import torch.utils.benchmark as benchmark
import pickle

d_dtype = {
    'half': torch.half,
    'float': torch.float
}

def prof(dtype, op, nl, hidden_size_max):
    fuzzer = benchmark.Fuzzer(
        parameters=[
            benchmark.FuzzedParameter('s', minval=1000, maxval=6000, distribution='uniform'),    # seq_length
            benchmark.FuzzedParameter('b', minval=1, maxval=64, distribution='uniform'),   # batch_size
            benchmark.FuzzedParameter('i', minval=16, maxval=512, distribution='uniform'),   # input_size
            benchmark.FuzzedParameter('h', minval=16, maxval=hidden_size_max, distribution='uniform'),   # hidden_size
            benchmark.FuzzedParameter('n', minval=1, maxval=4, distribution='uniform'),   # num_layer
        ],
        tensors=[
            benchmark.FuzzedTensor('x',
                                   size='sbi',
                                   min_elements=12,
                                   max_elements=10000000,
                                   cuda=True,
                                   dtype=d_dtype[dtype],
                                   max_allocation_bytes=1_000_000_000)
        ],
        seed=42,
        constraints=[
            lambda params: params['i'] % 8 == 0,
            lambda params: params['h'] % 8 == 0
        ])

    res = []

    for tensors, tensor_params, params in fuzzer.take(20):
        s = params['s']
        b = params['b']
        i = params['i']
        h = params['h']
        n = params['n']
        sub_label = f'x=({s}, {b}, {i}),'.ljust(20) + f'op=({i}, {h}, {n})'
        # sub_label = str(tensors['x'].size())

        if nl is None:
            setup=f'rnn=torch.nn.{op}({i}, {h}, {n})'
        else:
            setup=f'rnn=torch.nn.{op}({i}, {h}, {n}, nonlinearity="{nl}")'
        setup += f'.to(device="cuda", dtype={d_dtype[dtype]});'
        setup += 'out=rnn(x)[0]; g=torch.randn_like(out); '

        res.append(
            benchmark.Timer(stmt=f'out.backward(g, retain_graph=True)',
                            setup=setup,
                            globals=tensors,
                            label=f"{op=}, nonlinearity='{nl}', {dtype=}",
                            sub_label=sub_label,
                            description=f'{torch.__version__}')
                        .blocked_autorange(min_run_time=0.1))

    torch_ver = str(torch.__version__)
    torch_git_ver = torch_ver[torch_ver.index('+') + 1:]

    with open(f'{torch_git_ver}-{op}-{nl}-{dtype}.pkl', 'wb') as f:
        pickle.dump(res, f)

    compare = benchmark.Compare(res)
    # compare.colorize()
    compare.print()

for dtype in ['half', 'float']:
    for nl in ['tanh', 'relu']:
        prof(dtype, 'RNN', nl, 384)
    prof(dtype, 'LSTM', None, 192)
