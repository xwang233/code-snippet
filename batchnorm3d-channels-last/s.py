import torch
import time
import gc
import json
import itertools

niter = 100

dt = torch.half
de = 'cuda:0'
mf = torch.channels_last_3d
# Net = torch.nn.InstanceNorm3d
Net = torch.nn.BatchNorm3d

dd = itertools.product(
    [1, 2, 4, 8, 16],  # batch
    [2, 4, 8, 16, 32, 64],  # channel
    [2, 4, 8, 16, 32, 64, 112],  # dhw
)

j = {}
torchver = torch.__version__[8:]

j['torch-ver'] = torchver
j['cudnn-ver'] = str(torch.backends.cudnn.version())
j['device'] = torch.cuda.get_device_name(de)
j['dtype'] = str(dt)
j['memory_format'] = str(mf)
j['operator'] = str(Net)

print(json.dumps(j, indent=2))

def get_shape(d):
    shape = (d[0], d[1]) + (d[2], ) * 3
    return shape

def write_shape(shape, time_contiguous, time_channels_last):
    j[str(shape)] = {
        'contiguous-forward': time_contiguous[0],
        'contiguous-backward': time_contiguous[1],
        'channels-last-3d-forward': time_channels_last[0],
        'channels-last-3d-backward': time_channels_last[1]
    }

def write_message(shape, msg_key, msg_value):
    assert str(shape) in j
    j[str(shape)][msg_key] = msg_value

def profile(d):
    # i = 10

    shape = get_shape(d)

    print(f'{shape = }')

    x = torch.randn(*shape, dtype=dt,
                    device=de).to(memory_format=mf).requires_grad_()
    ref_x = x.detach().clone().contiguous().requires_grad_()

    # print(x.size(), x.stride())
    # print(ref_x.size(), ref_x.stride())

    c = x.shape[1]

    conv = Net(c)
    # conv = conv.to(memory_format=mf, device=de, dtype=dt)
    conv = conv.to(memory_format=mf, device=de, dtype=torch.float)
    ref_conv = Net(c)
    ref_conv.load_state_dict(conv.state_dict())
    # ref_conv = ref_conv.to(memory_format=torch.contiguous_format, device=de, dtype=dt)
    ref_conv = ref_conv.to(memory_format=torch.contiguous_format,
                           device=de,
                           dtype=torch.float)

    # if not conv.weight.is_contiguous(memory_format=mf):
    #     print('conv.weight is not in the correct memory format!')
    # print(conv.weight.size(), conv.weight.stride())
    # print(ref_conv.weight.size(), ref_conv.weight.stride())

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    time_contiguous = []
    time_channels_last = []

    def profile(p):
        start = time.time()
        for _ in range(niter):
            p()
        torch.cuda.synchronize()
        end = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return (end - start) / niter * 1000

    oom = False
    try:
        with torch.backends.cudnn.flags(enabled=True,
                                        benchmark=False,
                                        deterministic=False):
            out = conv(x)
            ref_out = ref_conv(ref_x)

            if not out.is_contiguous(memory_format=mf):
                raise RuntimeError('out is not in the correct memory format!')

            go = torch.randn_like(out).to(memory_format=mf,
                                          device=de,
                                          dtype=dt)
            ref_go = torch.randn_like(ref_out)
            for _ in range(niter):
                out = conv(x)
                ref_out = ref_conv(ref_x)
                out.backward(go, retain_graph=True)
                ref_out.backward(ref_go, retain_graph=True)

            if not x.grad.is_contiguous(memory_format=mf):
                raise RuntimeError(
                    'input grad is not in the correct memory format!')

            # if not conv.weight.grad.is_contiguous(memory_format=mf):
            #     print('weight grad is not in the correct memory format!')

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            time_channels_last += [
                profile(lambda: conv(x)),
                profile(lambda: out.backward(go, retain_graph=True))
            ]
            time_contiguous += [
                profile(lambda: ref_conv(ref_x)),
                profile(lambda: ref_out.backward(ref_go, retain_graph=True))
            ]

            torch.cuda.profiler.start()
            out = conv(x)
            ref_out = ref_conv(ref_x)
            out.backward(go, retain_graph=True)
            ref_out.backward(ref_go, retain_graph=True)
            torch.cuda.profiler.stop()
    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory'):
            oom = True
        else:
            raise

    torch.cuda.synchronize()

    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("OOM, continue")
        return

    # print(out.size(), out.stride())
    # print(ref_out.size(), ref_out.stride())

    print('contiguous', time_contiguous)
    print('channels_last', time_channels_last)

    write_shape(shape, time_contiguous, time_channels_last)

    try:
        _a = None
        _b = None
        _a, _b = torch.testing._core._compare_tensors_internal(out,
                                                               ref_out,
                                                               rtol=1e-5,
                                                               atol=2e-5,
                                                               equal_nan=False)
        # assert _a, _b
    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory'):
            pass
        else:
            raise
    finally:
        if _a is False:
            print(_b)
            write_message(shape, 'mismatch', _b)

    print()

for d in dd:
    oom = False
    try:
        profile(d)
    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory'):
            oom = True
        else:
            raise
    finally:
        shape = str(get_shape(d))
        if shape not in j:
            write_shape(shape, [-1, -1], [-1, -1])

    torch.cuda.synchronize()

    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("OOM, continue")

with open(f'res-{torchver}.json', 'w') as fj:
    json.dump(j, fj, indent=4)
