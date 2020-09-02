| shape | time_before (ms) | time_after (ms) |
| --- | --- | --- | 
| (2, 3, 4, 4), torch.contiguous_format, cpu  |  0.035 |  0.035 | 
| (2, 3, 4, 4), torch.contiguous_format, cuda  |  0.041 |  0.038 | 
| (2, 3, 4, 4), torch.channels_last, cpu  |  0.027 |  0.031 | 
| (2, 3, 4, 4), torch.channels_last, cuda  |  0.031 |  0.039 | 
| (2, 3, 4, 4), non_contiguous, cpu  |  0.037 |  0.032 | 
| (2, 3, 4, 4), non_contiguous, cuda  |  0.062 |  0.039 | 
| (4, 16, 32, 32), torch.contiguous_format, cpu  |  0.063 |  0.049 | 
| (4, 16, 32, 32), torch.contiguous_format, cuda  |  0.043 |  0.039 | 
| (4, 16, 32, 32), torch.channels_last, cpu  |  0.052 |  0.060 | 
| (4, 16, 32, 32), torch.channels_last, cuda  |  0.190 |  0.039 | 
| (4, 16, 32, 32), non_contiguous, cpu  |  0.048 |  0.040 | 
| (4, 16, 32, 32), non_contiguous, cuda  |  0.062 |  0.039 | 
| (8, 128, 64, 64), torch.contiguous_format, cpu  |  0.120 |  0.102 | 
| (8, 128, 64, 64), torch.contiguous_format, cuda  |  0.043 |  0.043 | 
| (8, 128, 64, 64), torch.channels_last, cpu  |  1.303 |  0.254 | 
| (8, 128, 64, 64), torch.channels_last, cuda  |  1.237 |  0.049 | 
| (8, 128, 64, 64), non_contiguous, cpu  |  0.132 |  0.137 | 
| (8, 128, 64, 64), non_contiguous, cuda  |  0.062 |  0.039 | 
| (16, 256, 224, 224), torch.contiguous_format, cpu  |  17.232 |  15.113 | 
| (16, 256, 224, 224), torch.contiguous_format, cuda  |  1.930 |  1.930 | 
| (16, 256, 224, 224), torch.channels_last, cpu  |  245.025 |  21.962 | 
| (16, 256, 224, 224), torch.channels_last, cuda  |  15.593 |  1.943 | 
| (16, 256, 224, 224), non_contiguous, cpu  |  11.738 |  4.933 | 
| (16, 256, 224, 224), non_contiguous, cuda  |  0.524 |  0.252 | 
