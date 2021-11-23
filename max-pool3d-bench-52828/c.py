import torch.utils.benchmark as benchmark
import glob
import pickle

res = []

for pkl in glob.glob('./*.pkl'):
    with open(pkl, 'rb') as f:
        res += pickle.load(f)

compare = benchmark.Compare(res)
compare.print()
