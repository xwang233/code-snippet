import torch
import subprocess

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


for i in range(len(dd)):
    d = dd[i][2]
    shape = [*dd[i][:2], d, d, d]
    oc = dd[i][3]
    ks = dd[i][4]
    pad = dd[i][5]
    stride = dd[i][6]

    # print(dd[i])
    print(f'shape {shape}, out_channel {oc}, kernel_size {ks}, padding {pad}, conv_stride {stride}')

    shape_str = ','.join(str(x) for x in shape)

    p = subprocess.run(['nsys', 'nvprof', '--profile-from-start', 'off', 'python', '-c',
rf'''
import torch
print(torch.__version__)

x = torch.randn({shape_str}, dtype=torch.half, device='cuda').to(memory_format=torch.channels_last_3d)
net = torch.nn.Conv3d({shape[1]}, {oc}, kernel_size={ks}, padding={pad}, stride={stride})
net = net.to(memory_format=torch.channels_last_3d, dtype=torch.half, device='cuda')
out = net(x)
g = torch.randn_like(out)

torch.cuda.synchronize()
torch.cuda.profiler.start()
for _ in range(200):
    out = net(x)
    out.backward(g, retain_graph=True)

torch.cuda.synchronize()
torch.cuda.profiler.stop()

'''], stdout=subprocess.PIPE)

    pso: str = p.stdout.decode('utf-8')

    for line in pso.splitlines():
        pass

