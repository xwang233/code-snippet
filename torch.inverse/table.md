| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.066 |  7.318 |  0.162  | 
| [] 4 torch.float32 |  0.011 |  7.370 |  0.160  | 
| [] 8 torch.float32 |  0.012 |  7.376 |  0.161  | 
| [] 16 torch.float32 |  0.028 |  7.359 |  0.135  | 
| [] 32 torch.float32 |  0.068 |  7.437 |  0.187  | 
| [] 64 torch.float32 |  0.111 |  7.568 |  0.270  | 
| [] 128 torch.float32 |  0.413 |  7.896 |  0.488  | 
| [] 256 torch.float32 |  1.117 |  11.620 |  1.082  | 
| [] 512 torch.float32 |  5.159 |  14.830 |  2.582  | 
| [] 1024 torch.float32 |  18.305 |  18.580 |  7.057  | 
| [1] 2 torch.float32 |  0.009 |  0.110 |  0.161 ***regressed | 
| [1] 4 torch.float32 |  0.009 |  0.113 |  0.170 ***regressed | 
| [1] 8 torch.float32 |  0.012 |  0.113 |  0.143 ***regressed | 
| [1] 16 torch.float32 |  0.016 |  0.119 |  0.139 ***regressed | 
| [1] 32 torch.float32 |  0.032 |  0.196 |  0.177  | 
| [1] 64 torch.float32 |  0.070 |  0.422 |  0.272  | 
| [1] 128 torch.float32 |  0.306 |  0.803 |  0.500  | 
| [1] 256 torch.float32 |  1.098 |  1.684 |  1.078  | 
| [1] 512 torch.float32 |  4.647 |  4.262 |  2.605  | 
| [1] 1024 torch.float32 |  16.645 |  16.350 |  6.934  | 
| [2] 2 torch.float32 |  0.009 |  0.111 |  0.191 ***regressed | 
| [2] 4 torch.float32 |  0.010 |  0.114 |  0.182 ***regressed | 
| [2] 8 torch.float32 |  0.012 |  0.115 |  0.183 ***regressed | 
| [2] 16 torch.float32 |  0.020 |  0.119 |  0.172 ***regressed | 
| [2] 32 torch.float32 |  0.051 |  0.170 |  0.241 ***regressed | 
| [2] 64 torch.float32 |  0.120 |  0.430 |  0.366  | 
| [2] 128 torch.float32 |  0.618 |  0.909 |  0.672  | 
| [2] 256 torch.float32 |  2.045 |  1.854 |  1.451  | 
| [2] 512 torch.float32 |  8.871 |  4.579 |  3.543  | 
| [2] 1024 torch.float32 |  34.065 |  18.420 |  12.160  | 
| [4] 2 torch.float32 |  0.010 |  0.111 |  0.328 ***regressed | 
| [4] 4 torch.float32 |  0.010 |  0.114 |  0.327 ***regressed | 
| [4] 8 torch.float32 |  0.014 |  0.116 |  0.329 ***regressed | 
| [4] 16 torch.float32 |  0.027 |  0.120 |  0.337 ***regressed | 
| [4] 32 torch.float32 |  0.085 |  0.174 |  0.393 ***regressed | 
| [4] 64 torch.float32 |  0.221 |  0.439 |  0.649 ***regressed | 
| [4] 128 torch.float32 |  1.083 |  0.860 |  1.219 ***regressed | 
| [4] 256 torch.float32 |  4.112 |  1.846 |  2.016 ***regressed | 
| [4] 512 torch.float32 |  18.680 |  5.138 |  5.079  | 
| [4] 1024 torch.float32 |  69.250 |  20.070 |  19.970  | 
