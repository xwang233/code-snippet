| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.010 |  7.446 |  0.117  | 
| [] 4 torch.float32 |  0.010 |  7.427 |  0.125  | 
| [] 8 torch.float32 |  0.011 |  7.571 |  0.127  | 
| [] 16 torch.float32 |  0.016 |  7.522 |  0.125  | 
| [] 32 torch.float32 |  0.033 |  7.548 |  0.173  | 
| [] 64 torch.float32 |  0.072 |  7.708 |  0.270  | 
| [] 128 torch.float32 |  0.352 |  8.024 |  0.481  | 
| [] 256 torch.float32 |  1.107 |  11.340 |  1.081  | 
| [] 512 torch.float32 |  4.985 |  15.010 |  2.581  | 
| [] 1024 torch.float32 |  19.390 |  19.270 |  6.952  | 
| [1] 2 torch.float32 |  0.009 |  0.114 |  0.134 ***regressed | 
| [1] 4 torch.float32 |  0.009 |  0.117 |  0.133 ***regressed | 
| [1] 8 torch.float32 |  0.011 |  0.126 |  0.132 ***regressed | 
| [1] 16 torch.float32 |  0.016 |  0.127 |  0.130 ***regressed | 
| [1] 32 torch.float32 |  0.032 |  0.178 |  0.174  | 
| [1] 64 torch.float32 |  0.070 |  0.420 |  0.268  | 
| [1] 128 torch.float32 |  0.318 |  0.801 |  0.503  | 
| [1] 256 torch.float32 |  1.058 |  1.674 |  1.080  | 
| [1] 512 torch.float32 |  4.692 |  4.791 |  2.585  | 
| [1] 1024 torch.float32 |  16.600 |  16.270 |  6.948  | 
| [2] 2 torch.float32 |  0.010 |  0.114 |  0.189 ***regressed | 
| [2] 4 torch.float32 |  0.010 |  0.120 |  0.184 ***regressed | 
| [2] 8 torch.float32 |  0.012 |  0.118 |  0.184 ***regressed | 
| [2] 16 torch.float32 |  0.020 |  0.129 |  0.176 ***regressed | 
| [2] 32 torch.float32 |  0.051 |  0.173 |  0.245 ***regressed | 
| [2] 64 torch.float32 |  0.120 |  0.427 |  0.376  | 
| [2] 128 torch.float32 |  0.653 |  0.914 |  0.729  | 
| [2] 256 torch.float32 |  2.152 |  1.799 |  1.477  | 
| [2] 512 torch.float32 |  9.358 |  4.506 |  3.555  | 
| [2] 1024 torch.float32 |  32.910 |  18.170 |  12.330  | 
| [4] 2 torch.float32 |  0.010 |  0.115 |  0.324 ***regressed | 
| [4] 4 torch.float32 |  0.010 |  0.118 |  0.324 ***regressed | 
| [4] 8 torch.float32 |  0.014 |  0.118 |  0.324 ***regressed | 
| [4] 16 torch.float32 |  0.028 |  0.122 |  0.333 ***regressed | 
| [4] 32 torch.float32 |  0.088 |  0.175 |  0.392 ***regressed | 
| [4] 64 torch.float32 |  0.225 |  0.431 |  0.650 ***regressed | 
| [4] 128 torch.float32 |  1.140 |  0.932 |  1.048 ***regressed | 
| [4] 256 torch.float32 |  4.153 |  1.808 |  2.034 ***regressed | 
| [4] 512 torch.float32 |  18.280 |  4.896 |  5.069 ***regressed | 
| [4] 1024 torch.float32 |  62.975 |  19.990 |  19.700  | 
