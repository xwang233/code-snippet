import torch.utils.benchmark as benchmark
import glob
import pickle

res = []

for pkl in sorted(glob.glob('./*.pkl')):
    with open(pkl, 'rb') as f:
        res += pickle.load(f)

compare = benchmark.Compare(res)
compare.print()
