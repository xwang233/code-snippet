| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.015 |  7.446 |  0.127  | 
| [] 4 torch.float32 |  0.010 |  7.427 |  0.118  | 
| [] 8 torch.float32 |  0.012 |  7.571 |  0.127  | 
| [] 16 torch.float32 |  0.016 |  7.522 |  0.123  | 
| [] 32 torch.float32 |  0.033 |  7.548 |  0.175  | 
| [] 64 torch.float32 |  0.073 |  7.708 |  0.256  | 
| [] 128 torch.float32 |  0.363 |  8.024 |  0.478  | 
| [] 256 torch.float32 |  1.094 |  11.340 |  1.053  | 
| [] 512 torch.float32 |  5.094 |  15.010 |  2.559  | 
| [] 1024 torch.float32 |  18.715 |  19.270 |  6.929  | 
| [1] 2 torch.float32 |  0.009 |  0.114 |  0.146 ***regressed | 
| [1] 4 torch.float32 |  0.010 |  0.117 |  0.145 ***regressed | 
| [1] 8 torch.float32 |  0.011 |  0.126 |  0.145 ***regressed | 
| [1] 16 torch.float32 |  0.017 |  0.127 |  0.134 ***regressed | 
| [1] 32 torch.float32 |  0.033 |  0.178 |  0.178 ***regressed | 
| [1] 64 torch.float32 |  0.071 |  0.420 |  0.276  | 
| [1] 128 torch.float32 |  0.316 |  0.801 |  0.494  | 
| [1] 256 torch.float32 |  1.072 |  1.674 |  1.071  | 
| [1] 512 torch.float32 |  4.667 |  4.791 |  2.573  | 
| [1] 1024 torch.float32 |  16.995 |  16.270 |  7.023  | 
| [2] 2 torch.float32 |  0.012 |  0.114 |  0.236 ***regressed | 
| [2] 4 torch.float32 |  0.012 |  0.120 |  0.235 ***regressed | 
| [2] 8 torch.float32 |  0.015 |  0.118 |  0.235 ***regressed | 
| [2] 16 torch.float32 |  0.024 |  0.129 |  0.218 ***regressed | 
| [2] 32 torch.float32 |  0.055 |  0.173 |  0.289 ***regressed | 
| [2] 64 torch.float32 |  0.129 |  0.427 |  0.418  | 
| [2] 128 torch.float32 |  0.613 |  0.914 |  0.682  | 
| [2] 256 torch.float32 |  2.067 |  1.799 |  1.511  | 
| [2] 512 torch.float32 |  9.079 |  4.506 |  3.555  | 
| [2] 1024 torch.float32 |  33.390 |  18.170 |  12.460  | 
| [4] 2 torch.float32 |  0.010 |  0.115 |  0.332 ***regressed | 
| [4] 4 torch.float32 |  0.011 |  0.118 |  0.346 ***regressed | 
| [4] 8 torch.float32 |  0.014 |  0.118 |  0.334 ***regressed | 
| [4] 16 torch.float32 |  0.028 |  0.122 |  0.337 ***regressed | 
| [4] 32 torch.float32 |  0.086 |  0.175 |  0.390 ***regressed | 
| [4] 64 torch.float32 |  0.226 |  0.431 |  0.673 ***regressed | 
| [4] 128 torch.float32 |  1.111 |  0.932 |  1.103 ***regressed | 
| [4] 256 torch.float32 |  3.954 |  1.808 |  2.162 ***regressed | 
| [4] 512 torch.float32 |  18.265 |  4.896 |  5.526 ***regressed | 
| [4] 1024 torch.float32 |  61.895 |  19.990 |  21.460 ***regressed | 
