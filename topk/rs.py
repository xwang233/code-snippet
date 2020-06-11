import torch
print([int(1e4), int(5e4), int(1e5), int(5e5), int(1e6), int(5e6), int(1e7)] \
        + torch.randint(int(1e5), int(5e6), (7, )).tolist())