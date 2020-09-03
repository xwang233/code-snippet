| shape | time_before (ms) | time_after (ms) |
| --- | --- | --- | 
| (2, 3, 4, 4), torch.contiguous_format, cpu  |  0.035 |  0.031 | 
| (2, 3, 4, 4), torch.contiguous_format, cuda  |  0.041 |  0.031 | 
| (2, 3, 4, 4), torch.channels_last, cpu  |  0.027 |  0.029 | 
| (2, 3, 4, 4), torch.channels_last, cuda  |  0.031 |  0.034 | 
| (2, 3, 4, 4), non_contiguous, cpu  |  0.037 |  0.026 | 
| (2, 3, 4, 4), non_contiguous, cuda  |  0.062 |  0.033 | 
| (4, 16, 32, 32), torch.contiguous_format, cpu  |  0.063 |  0.055 | 
| (4, 16, 32, 32), torch.contiguous_format, cuda  |  0.043 |  0.031 | 
| (4, 16, 32, 32), torch.channels_last, cpu  |  0.052 |  0.064 | 
| (4, 16, 32, 32), torch.channels_last, cuda  |  0.190 |  0.033 | 
| (4, 16, 32, 32), non_contiguous, cpu  |  0.048 |  0.035 | 
| (4, 16, 32, 32), non_contiguous, cuda  |  0.062 |  0.033 | 
| (8, 128, 64, 64), torch.contiguous_format, cpu  |  0.120 |  0.109 | 
| (8, 128, 64, 64), torch.contiguous_format, cuda  |  0.043 |  0.044 | 
| (8, 128, 64, 64), torch.channels_last, cpu  |  1.303 |  0.260 | 
| (8, 128, 64, 64), torch.channels_last, cuda  |  1.237 |  0.049 | 
| (8, 128, 64, 64), non_contiguous, cpu  |  0.132 |  0.128 | 
| (8, 128, 64, 64), non_contiguous, cuda  |  0.062 |  0.031 | 
| (16, 256, 224, 224), torch.contiguous_format, cpu  |  17.232 |  14.807 | 
| (16, 256, 224, 224), torch.contiguous_format, cuda  |  1.930 |  1.930 | 
| (16, 256, 224, 224), torch.channels_last, cpu  |  245.025 |  24.345 | 
| (16, 256, 224, 224), torch.channels_last, cuda  |  15.593 |  1.944 | 
| (16, 256, 224, 224), non_contiguous, cpu  |  11.738 |  6.460 | 
| (16, 256, 224, 224), non_contiguous, cuda  |  0.524 |  0.251 | 
