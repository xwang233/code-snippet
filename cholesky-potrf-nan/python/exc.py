import torch
from array import array

print(torch.__version__)

x = torch.load('cholesky_cuda.pt')
# x.shape = [16, 256, 256]
# x.dtype = torch.float
# x.device = 'cuda'

print('x contains nan?', x.isnan().any().item())

print(  torch.cholesky(x.cpu())[2]  )   # CPU path

print(  torch.cholesky(x[2])  )     # GPU path, take the 2nd slice, call cusolver potrf

print(  torch.cholesky(x)[2]  )     # GPU path, call cusolver potrf batched, take the 2nd slice

print(  torch.cholesky(x.double())[2]  )    
                            # GPU path, convert to double, call cusolver potrf batched,
                            # take the 2nd slice

xfc = x.flatten().cpu()

print(xfc[:10])
print(xfc[-10:])

with open('.data.bin', 'wb') as f:
    ax = array('f', xfc.numpy().tolist())
    ax.tofile(f)

## This loop can also trigger a nan exception in about 5 ~ 10 random attempts
# for i in range(1000):
#     try:
#         r = torch.randn(16, 256, 256, dtype=torch.float, device='cuda')
#         x_  = torch.matmul(r, r.transpose(-1, -2))
# 
#         # x = x_[2]
#         x = x_
# 
#         # print(x.size())
# 
#         y = torch.cholesky(x)
# 
#         torch.cuda.synchronize()
#     except:
#         print(i)
#         raise

