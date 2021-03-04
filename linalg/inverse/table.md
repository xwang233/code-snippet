| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.095 |  7.534 |  0.129  | 
| [] 4 torch.float32 |  0.009 |  7.522 |  0.129  | 
| [] 8 torch.float32 |  0.011 |  7.647 |  0.138  | 
| [] 16 torch.float32 |  0.075 |  7.582 |  0.135  | 
| [] 32 torch.float32 |  0.073 |  7.573 |  0.191  | 
| [] 64 torch.float32 |  0.134 |  7.694 |  0.288  | 
| [] 128 torch.float32 |  0.398 |  8.073 |  0.491  | 
| [] 256 torch.float32 |  1.054 |  11.860 |  1.074  | 
| [] 512 torch.float32 |  5.218 |  14.130 |  2.582  | 
| [] 1024 torch.float32 |  19.010 |  18.780 |  6.936  | 
| [1] 2 torch.float32 |  0.009 |  0.113 |  0.128 ***regressed | 
| [1] 4 torch.float32 |  0.009 |  0.113 |  0.131 ***regressed | 
| [1] 8 torch.float32 |  0.011 |  0.116 |  0.129 ***regressed | 
| [1] 16 torch.float32 |  0.015 |  0.122 |  0.135 ***regressed | 
| [1] 32 torch.float32 |  0.032 |  0.177 |  0.178 ***regressed | 
| [1] 64 torch.float32 |  0.070 |  0.420 |  0.281  | 
| [1] 128 torch.float32 |  0.328 |  0.816 |  0.490  | 
| [1] 256 torch.float32 |  1.125 |  1.690 |  1.084  | 
| [1] 512 torch.float32 |  4.344 |  4.305 |  2.576  | 
| [1] 1024 torch.float32 |  16.510 |  16.340 |  6.928  | 
| [2] 2 torch.float32 |  0.009 |  0.113 |  0.186 ***regressed | 
| [2] 4 torch.float32 |  0.011 |  0.115 |  0.184 ***regressed | 
| [2] 8 torch.float32 |  0.012 |  0.114 |  0.184 ***regressed | 
| [2] 16 torch.float32 |  0.019 |  0.119 |  0.173 ***regressed | 
| [2] 32 torch.float32 |  0.050 |  0.170 |  0.240 ***regressed | 
| [2] 64 torch.float32 |  0.120 |  0.429 |  0.375  | 
| [2] 128 torch.float32 |  0.576 |  0.830 |  0.675  | 
| [2] 256 torch.float32 |  2.021 |  1.748 |  1.451  | 
| [2] 512 torch.float32 |  9.070 |  4.749 |  3.539  | 
| [2] 1024 torch.float32 |  33.655 |  18.240 |  12.220  | 
| [4] 2 torch.float32 |  0.009 |  0.112 |  0.318 ***regressed | 
| [4] 4 torch.float32 |  0.010 |  0.115 |  0.319 ***regressed | 
| [4] 8 torch.float32 |  0.013 |  0.115 |  0.320 ***regressed | 
| [4] 16 torch.float32 |  0.027 |  0.120 |  0.331 ***regressed | 
| [4] 32 torch.float32 |  0.085 |  0.173 |  0.385 ***regressed | 
| [4] 64 torch.float32 |  0.221 |  0.431 |  0.646 ***regressed | 
| [4] 128 torch.float32 |  1.102 |  0.834 |  1.055 ***regressed | 
| [4] 256 torch.float32 |  4.042 |  1.811 |  2.054 ***regressed | 
| [4] 512 torch.float32 |  18.390 |  4.884 |  5.087 ***regressed | 
| [4] 1024 torch.float32 |  69.025 |  19.840 |  20.000 ***regressed | 
