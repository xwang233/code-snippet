import torch
import os
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt

from torch.testing._core import _compare_tensors_internal

DTYPE_PRECISIONS = torch.testing._asserts._DTYPE_PRECISIONS

commits = {
    'master': 'git6bb33d9',
    'PR':     'giteddb291'
}

d = {}
data_file_keys = set()

d_perf = {}

with open('shapes.json', 'r') as f:
    d_shapes = json.load(f)

for commit in commits:
    with open(f'perf-{commits[commit]}.json', 'r') as f:
        d_perf[commit] = json.load(f)

    data_path = f'data-{commits[commit]}'

    d[commit] = {}

    data_files = os.listdir(data_path)
    data_file_keys |= set(data_files)

    for data_file in data_files:
        d[commit][data_file] = torch.load(f'{data_path}/{data_file}')

not_match = 0

for data_file in data_file_keys:

    rtol, atol = DTYPE_PRECISIONS[
        d['master'][data_file].dtype
    ]

    # match: bool, messages_for_mismatch: str
    _a, _b = _compare_tensors_internal(
        d['master'][data_file],
        d['PR'][data_file],
        rtol=rtol, atol=atol, equal_nan=False)

    if _a:
        continue

    not_match += 1
    print('not match', data_file, _b)

if not_match == 0:
    print('all match')
else:
    print(f'found {not_match} not matched out of total {len(data_file_keys)}')

l_perf_compare = []

print('idx, time_master (us), time_pr (us), speed_up, shape')
for idx in d_perf['master']:
    perf_pr = d_perf['PR'][idx]
    perf_master = d_perf['master'][idx]

    perf_x = perf_master / perf_pr
    if perf_x > 1.05 or perf_x < 0.95:
        print(f'{idx: >3}, {perf_master: 11.3f}, {perf_pr : 11.3f}, {perf_x : 7.3f}, {d_shapes[idx]}')
    l_perf_compare.append(perf_master / perf_pr)

sns.displot(l_perf_compare)
plt.savefig('a.png')
