| shape | time_before (ms) | time_after (ms) |
| --- | --- | --- | 
| (2, 3, 4, 4), torch.contiguous_format, cpu  |  0.035 |  0.039 | 
| (2, 3, 4, 4), torch.contiguous_format, cuda  |  0.041 |  0.037 | 
| (2, 3, 4, 4), torch.channels_last, cpu  |  0.027 |  0.031 | 
| (2, 3, 4, 4), torch.channels_last, cuda  |  0.031 |  0.036 | 
| (2, 3, 4, 4), non_contiguous, cpu  |  0.037 |  0.031 | 
| (2, 3, 4, 4), non_contiguous, cuda  |  0.062 |  0.037 | 
| (4, 16, 32, 32), torch.contiguous_format, cpu  |  0.063 |  0.060 | 
| (4, 16, 32, 32), torch.contiguous_format, cuda  |  0.043 |  0.037 | 
| (4, 16, 32, 32), torch.channels_last, cpu  |  0.052 |  0.069 | 
| (4, 16, 32, 32), torch.channels_last, cuda  |  0.190 |  0.037 | 
| (4, 16, 32, 32), non_contiguous, cpu  |  0.048 |  0.041 | 
| (4, 16, 32, 32), non_contiguous, cuda  |  0.062 |  0.037 | 
| (8, 128, 64, 64), torch.contiguous_format, cpu  |  0.120 |  0.102 | 
| (8, 128, 64, 64), torch.contiguous_format, cuda  |  0.043 |  0.048 | 
| (8, 128, 64, 64), torch.channels_last, cpu  |  1.303 |  0.258 | 
| (8, 128, 64, 64), torch.channels_last, cuda  |  1.237 |  0.050 | 
| (8, 128, 64, 64), non_contiguous, cpu  |  0.132 |  0.136 | 
| (8, 128, 64, 64), non_contiguous, cuda  |  0.062 |  0.038 | 
| (16, 256, 224, 224), torch.contiguous_format, cpu  |  17.232 |  15.652 | 
| (16, 256, 224, 224), torch.contiguous_format, cuda  |  1.930 |  1.931 | 
| (16, 256, 224, 224), torch.channels_last, cpu  |  245.025 |  26.951 | 
| (16, 256, 224, 224), torch.channels_last, cuda  |  15.593 |  1.943 | 
| (16, 256, 224, 224), non_contiguous, cpu  |  11.738 |  6.370 | 
| (16, 256, 224, 224), non_contiguous, cuda  |  0.524 |  0.252 | 
