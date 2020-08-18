| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.016 |  7.446 |  0.168  | 
| [] 4 torch.float32 |  0.012 |  7.427 |  0.150  | 
| [] 8 torch.float32 |  0.013 |  7.571 |  0.152  | 
| [] 16 torch.float32 |  0.018 |  7.522 |  0.154  | 
| [] 32 torch.float32 |  0.036 |  7.548 |  0.204  | 
| [] 64 torch.float32 |  0.078 |  7.708 |  0.292  | 
| [] 128 torch.float32 |  0.333 |  8.024 |  0.515  | 
| [] 256 torch.float32 |  1.100 |  11.340 |  1.095  | 
| [] 512 torch.float32 |  5.150 |  15.010 |  2.602  | 
| [] 1024 torch.float32 |  18.395 |  19.270 |  7.000  | 
| [1] 2 torch.float32 |  0.009 |  0.114 |  0.142 ***regressed | 
| [1] 4 torch.float32 |  0.009 |  0.117 |  0.142 ***regressed | 
| [1] 8 torch.float32 |  0.011 |  0.126 |  0.140 ***regressed | 
| [1] 16 torch.float32 |  0.016 |  0.127 |  0.130 ***regressed | 
| [1] 32 torch.float32 |  0.032 |  0.178 |  0.173  | 
| [1] 64 torch.float32 |  0.070 |  0.420 |  0.270  | 
| [1] 128 torch.float32 |  0.341 |  0.801 |  0.492  | 
| [1] 256 torch.float32 |  1.134 |  1.674 |  1.101  | 
| [1] 512 torch.float32 |  4.574 |  4.791 |  2.607  | 
| [1] 1024 torch.float32 |  15.825 |  16.270 |  6.999  | 
| [2] 2 torch.float32 |  0.010 |  0.114 |  0.193 ***regressed | 
| [2] 4 torch.float32 |  0.011 |  0.120 |  0.190 ***regressed | 
| [2] 8 torch.float32 |  0.012 |  0.118 |  0.200 ***regressed | 
| [2] 16 torch.float32 |  0.020 |  0.129 |  0.178 ***regressed | 
| [2] 32 torch.float32 |  0.051 |  0.173 |  0.246 ***regressed | 
| [2] 64 torch.float32 |  0.120 |  0.427 |  0.378  | 
| [2] 128 torch.float32 |  0.560 |  0.914 |  0.685  | 
| [2] 256 torch.float32 |  2.155 |  1.799 |  1.479  | 
| [2] 512 torch.float32 |  9.041 |  4.506 |  3.537  | 
| [2] 1024 torch.float32 |  31.535 |  18.170 |  12.500  | 
| [4] 2 torch.float32 |  0.010 |  0.115 |  0.321 ***regressed | 
| [4] 4 torch.float32 |  0.010 |  0.118 |  0.320 ***regressed | 
| [4] 8 torch.float32 |  0.014 |  0.118 |  0.321 ***regressed | 
| [4] 16 torch.float32 |  0.027 |  0.122 |  0.340 ***regressed | 
| [4] 32 torch.float32 |  0.085 |  0.175 |  0.378 ***regressed | 
| [4] 64 torch.float32 |  0.224 |  0.431 |  0.652 ***regressed | 
| [4] 128 torch.float32 |  1.061 |  0.932 |  1.062 ***regressed | 
| [4] 256 torch.float32 |  3.918 |  1.808 |  2.025 ***regressed | 
| [4] 512 torch.float32 |  17.865 |  4.896 |  5.074 ***regressed | 
| [4] 1024 torch.float32 |  59.825 |  19.990 |  19.830  | 
