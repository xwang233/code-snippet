import torch

dtype = torch.float

seq_length = 5
batch_size = 3

input_size = 10
hidden_size = 20
num_layers = 2


a = torch.randn(seq_length, batch_size, input_size, dtype=dtype, device='cuda')
rnn = torch.nn.RNN(input_size, hidden_size, num_layers).cuda()

output = rnn(a)
