| shape | cpu_time (ms) | gpu_time_before (magma) (ms) | gpu_time_after (ms) |
| --- | --- | --- | --- | 
| [] 2 torch.float32 |  0.066 |  7.318 |  0.124  | 
| [] 4 torch.float32 |  0.009 |  7.370 |  0.123  | 
| [] 8 torch.float32 |  0.011 |  7.376 |  0.136  | 
| [] 16 torch.float32 |  0.027 |  7.359 |  0.132  | 
| [] 32 torch.float32 |  0.067 |  7.437 |  0.185  | 
| [] 64 torch.float32 |  0.110 |  7.568 |  0.268  | 
| [] 128 torch.float32 |  0.408 |  7.896 |  0.491  | 
| [] 256 torch.float32 |  1.155 |  11.620 |  1.078  | 
| [] 512 torch.float32 |  5.128 |  14.830 |  2.636  | 
| [] 1024 torch.float32 |  19.135 |  18.580 |  7.172  | 
| [1] 2 torch.float32 |  0.009 |  0.110 |  0.130 ***regressed | 
| [1] 4 torch.float32 |  0.009 |  0.113 |  0.131 ***regressed | 
| [1] 8 torch.float32 |  0.011 |  0.113 |  0.133 ***regressed | 
| [1] 16 torch.float32 |  0.016 |  0.119 |  0.135 ***regressed | 
| [1] 32 torch.float32 |  0.032 |  0.196 |  0.181  | 
| [1] 64 torch.float32 |  0.070 |  0.422 |  0.277  | 
| [1] 128 torch.float32 |  0.340 |  0.803 |  0.500  | 
| [1] 256 torch.float32 |  1.095 |  1.684 |  1.091  | 
| [1] 512 torch.float32 |  4.707 |  4.262 |  2.600  | 
| [1] 1024 torch.float32 |  16.875 |  16.350 |  6.973  | 
| [2] 2 torch.float32 |  0.009 |  0.111 |  0.193 ***regressed | 
| [2] 4 torch.float32 |  0.010 |  0.114 |  0.188 ***regressed | 
| [2] 8 torch.float32 |  0.013 |  0.115 |  0.188 ***regressed | 
| [2] 16 torch.float32 |  0.020 |  0.119 |  0.177 ***regressed | 
| [2] 32 torch.float32 |  0.050 |  0.170 |  0.242 ***regressed | 
| [2] 64 torch.float32 |  0.119 |  0.430 |  0.371  | 
| [2] 128 torch.float32 |  0.633 |  0.909 |  0.674  | 
| [2] 256 torch.float32 |  2.086 |  1.854 |  1.459  | 
| [2] 512 torch.float32 |  8.702 |  4.579 |  3.653  | 
| [2] 1024 torch.float32 |  33.205 |  18.420 |  12.490  | 
| [4] 2 torch.float32 |  0.012 |  0.111 |  0.367 ***regressed | 
| [4] 4 torch.float32 |  0.012 |  0.114 |  0.370 ***regressed | 
| [4] 8 torch.float32 |  0.015 |  0.116 |  0.373 ***regressed | 
| [4] 16 torch.float32 |  0.030 |  0.120 |  0.377 ***regressed | 
| [4] 32 torch.float32 |  0.090 |  0.174 |  0.438 ***regressed | 
| [4] 64 torch.float32 |  0.229 |  0.439 |  0.725 ***regressed | 
| [4] 128 torch.float32 |  1.107 |  0.860 |  1.110 ***regressed | 
| [4] 256 torch.float32 |  4.204 |  1.846 |  2.197 ***regressed | 
| [4] 512 torch.float32 |  18.400 |  5.138 |  5.552 ***regressed | 
| [4] 1024 torch.float32 |  68.790 |  20.070 |  21.860 ***regressed | 
