import torch

import time

nb_iter = 1000

def p(device, dtype, msg):
    print(torch.__version__)
    print('device', device)

    fn = f'{msg}-{device}-{dtype}.txt'
    print(fn)

    with open(fn, 'w') as f:
        for c in [1, 10, 32]:
            for dhw in [7, 16, 32, 56, 64, 112, 100, 256]:
                for ks in [2, 3, 5, 7]:
                    torch.cuda.empty_cache()

                    x = torch.randn(1, c, dhw, dhw, dhw, device=device, dtype=dtype)
                    mp = torch.nn.MaxPool3d(ks)

                    # warmup
                    for _ in range(nb_iter):
                        out = mp(x)

                    torch.cuda.synchronize()
                    start_t = time.time()

                    for _ in range(nb_iter):
                        out = mp(x)

                    torch.cuda.synchronize()
                    end_t = time.time()

                    t = (end_t - start_t) / nb_iter

                    s = f'{c} {dhw} {ks} # {t}'
                    print(s)
                    f.write(s + '\n')

# p('cuda', torch.float16, 'before')
# p('cuda', torch.float32, 'before')

p('cuda', torch.float16, 'after')
p('cuda', torch.float32, 'after')