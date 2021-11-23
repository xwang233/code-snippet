import torch

x = torch.randn(70, 32, 100, 100, 100, dtype=torch.half, device='cuda')

y = torch.nn.functional.max_pool3d(x, 5)

torch.cuda.synchronize()