import torch
import time
import gc
import json

niter = 200

dt = torch.half
de = 'cuda:0'
mf = torch.channels_last_3d
Net = torch.nn.Conv3d

dd = [
    # (n, c, dhw, oc, ks, pad, stride)
    (2, 4, 128, 32, 3, 1, 1),
    (2, 32, 128, 32, 3, 1, 1),
    (2, 32, 128, 64, 3, 1, 2),
    (2, 64, 64, 64, 3, 1, 1),
    (2, 64, 64, 128, 3, 1, 2),
    (2, 128, 32, 128, 3, 1, 1),
    (2, 128, 32, 256, 3, 1, 2),
    (2, 256, 16, 256, 3, 1, 1),
    (2, 256, 16, 512, 3, 1, 2),
    (2, 512, 8, 512, 3, 1, 1),
    (2, 512, 8, 512, 3, 1, 2),
    (2, 512, 4, 512, 3, 1, 1),
    (2, 1024, 8, 512, 3, 1, 1),
    (2, 512, 16, 256, 3, 1, 1),
    (2, 256, 32, 128, 3, 1, 1),
    (2, 128, 64, 64, 3, 1, 1),
    (2, 64, 128, 32, 3, 1, 1),
    (2, 32, 128, 4, 1, 0, 1),
    (2, 32, 128, 64, 2, 0, 2),
    (2, 64, 64, 128, 2, 0, 2),
    (2, 128, 32, 256, 2, 0, 2),
    (2, 256, 16, 512, 2, 0, 2),
    (2, 512, 8, 512, 2, 0, 2)
]

j = {}
torchver = torch.__version__[8:]

j['torch-ver'] = torchver
j['cudnn-ver'] = str(torch.backends.cudnn.version())
j['device'] = torch.cuda.get_device_name(de)
j['dtype'] = str(dt)
j['memory_format'] = str(mf)
j['operator'] = str(Net)

print(json.dumps(j, indent=2))

for i in range(len(dd)):
    # i = 10
    d = dd[i][2]
    shape = [*dd[i][:2], d, d, d]
    oc = dd[i][3]
    ks = dd[i][4]
    pad = dd[i][5]
    stride = dd[i][6]

    # print(dd[i])
    print(f'shape {shape}, out_channel {oc}, kernel_size {ks}, padding {pad}, conv_stride {stride}')

    x = torch.randn(*shape, dtype=dt, device=de).to(memory_format=mf).requires_grad_()
    ref_x = x.detach().clone().contiguous().requires_grad_()

    # print(x.size(), x.stride())
    # print(ref_x.size(), ref_x.stride())

    c = x.shape[1]

    conv = Net(c, oc, kernel_size=ks, padding=pad, stride=stride)
    conv = conv.to(memory_format=mf, device=de, dtype=dt)
    ref_conv = Net(c, oc, kernel_size=ks, padding=pad, stride=stride)
    ref_conv.load_state_dict(conv.state_dict())
    ref_conv = ref_conv.to(memory_format=torch.contiguous_format, device=de, dtype=dt)

    if not conv.weight.is_contiguous(memory_format=mf):
        print('conv.weight is not in the correct memory format!')
    # print(conv.weight.size(), conv.weight.stride())
    # print(ref_conv.weight.size(), ref_conv.weight.stride())

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    time_contiguous = []
    time_channels_last = []

    def profile(p):
        start = time.time()
        for _ in range(niter): p()
        torch.cuda.synchronize()
        end = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return (end-start) / niter * 1000

    oom = False
    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False):
            out = conv(x)
            ref_out = ref_conv(ref_x)
            go = torch.randn_like(out).to(memory_format=mf, device=de, dtype=dt)
            ref_go = torch.randn_like(ref_out)
            for _ in range(niter):
                out = conv(x)
                ref_out = ref_conv(ref_x)
                out.backward(go, retain_graph=True)
                ref_out.backward(ref_go, retain_graph=True)
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

    # print(out.size(), out.stride())
    # print(ref_out.size(), ref_out.stride())

    print('contiguous', time_contiguous)
    print('channels_last', time_channels_last)
    if not out.is_contiguous(memory_format=mf):
        print('output is not in the correct memory format')

    print()

    j[str(dd[i])] = {
        'contiguous-forward': time_contiguous[0],
        'contiguous-backward': time_contiguous[1],
        'channels-last-3d-forward': time_channels_last[0],
        'channels-last-3d-backward': time_channels_last[1]
    }

    try:
        _a = None
        _b = None
        _a, _b = torch.testing._compare_tensors_internal(out, ref_out, rtol=1e-5, atol=2e-5, equal_nan=False)
        # assert _a, _b
    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory'):
            pass
        else:
            raise
    finally:
        if _a is False: print(_b)

with open(f'res-{torchver}.json', 'w') as fj:
    json.dump(j, fj, indent=4)