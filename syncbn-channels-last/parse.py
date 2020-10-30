import shutil
import glob
import json
import io
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

AFTER  = 'feabdaf'
BEFORE = '5e2f17d'

class Markdown:
    def __init__(self):
        self.buffer = io.BufferedRandom(io.BytesIO())
        self.enc = 'ascii'
    
    def write(self, s: str):
        self.buffer.write(s.encode(self.enc))
    
    def read(self) -> bytes:
        self.buffer.seek(0)
        return self.buffer.read()


def main():
    js = {}

    for file in glob.glob('output*.json'):
        commit = file[file.index('+')+1 : -5]

        with open(file, 'r') as f:
            j = json.load(f)
        
        js[commit] = j
    
    # print(js)

    columns = [
        'native_contiguous_before',
        'native_channels_last_after',
        'apex_channels_last',
        'native_channels_last_after vs native_contiguous_before',
        'native_channels_last_after vs apex_channels_last'
    ]

    # res -> shape -> column -> time
    res = defaultdict(lambda: defaultdict(int))
    for commit in js:
        for shape in js[commit]:
            for k1 in js[commit][shape]:
                for k2 in js[commit][shape][k1]:
                    # print(commit, shape, k1, k2)
                    v = js[commit][shape][k1][k2]

                    idx = -1
                    if 'batch_norm' in k2: # native
                        if 'channels_last' in k2 and commit == AFTER:
                            idx = 1
                        elif 'channels_last' not in k2 and commit == BEFORE:
                            idx = 0
                    else: # apex
                        if 'c_last' in k2 and commit == AFTER:
                            idx = 2
 
                    if idx >= 0:
                        res[shape][columns[idx]] += v / 1e6 if v > 0 else v
        

    print(json.dumps(res, indent=2))

    diff1 = []
    diff2 = []

    md = Markdown()
    md.write('before commit **' + BEFORE + '**\n\n')
    md.write('after commit **' + AFTER + '**\n\n')
    md.write('time is in **ms** (10^-3 s), negative percentage is better\n\n')
    md.write('|shape|' + '|'.join(columns) + '|\n')
    md.write('|---:' * (len(columns)+1) + '|\n')
    for shape in sorted(res.keys(), key=lambda x: eval(x)):
        md.write('|' + shape + '|' + '|'.join(f'{res[shape][s] : .3f}' for s in columns[:-2]) + '|')
        ncb  = res[shape][columns[0]]
        ncla = res[shape][columns[1]]
        acl  = res[shape][columns[2]]

        d1 = (ncla-ncb)/ncb*100
        d2 = (ncla-acl)/acl*100
        if d1 < 50:
            md.write(f'{d1 : .2f}% |')
        else:
            md.write(f'<div style="color:red;">{d1 : .2f}%</div> |')
        md.write(f'{d2 : .2f}% |\n')

        diff1.append(d1)
        diff2.append(d2)

    with open('readme.md', 'wb') as f:
        f.write(md.read())
    
    ax1 = sns.histplot(diff1, kde=True)
    plt.xlabel('time cost change (%), negative means better perf')
    plt.ylabel('PDF')
    # ax1 = sns.displot(diff1, kde=True)
    # ax1.set_axis_labels('perf change (%)', 'PDF')
    plt.tight_layout()
    plt.savefig('new_channels_last-vs-native_contiguous.png', transparent=True)
    plt.clf()

    ax2 = sns.histplot(diff2, kde=True)
    plt.xlabel('time cost change (%), negative means better perf')
    plt.ylabel('PDF')
    # ax2 = sns.displot(diff2, kde=True)
    # ax2.set_axis_labels('perf change (%)', 'PDF')
    plt.tight_layout()
    plt.savefig('new_channels_last-vs-apex_channels_last.png', transparent=True)
    plt.clf()


if __name__ == "__main__":
    main()