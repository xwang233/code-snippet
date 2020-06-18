import torch
print(torch.__version__)

sizes = [5 * int(1.1**i) for i in range(1000)]
i = 0
crash_count = 4

while i < len(sizes):
    size = sizes[i]

    try:
        torch.cuda.synchronize()

        x = torch.randn(1, 16, size, size, dtype=torch.float, device='cuda', requires_grad=True)
        op = torch.log
        y = op(x)

        g = torch.randn_like(y)
        y.backward(g)

        torch.cuda.synchronize()
    except RuntimeError as e:
        if not str(e).startswith('CUDA out of memory'):
            raise
        # only capture RuntimeError: CUDA out of memory

        torch.cuda.synchronize()
        print(size)

        crash_count -= 1
        if crash_count == 0:
            break
        
        i = 0
        continue

    i += 1