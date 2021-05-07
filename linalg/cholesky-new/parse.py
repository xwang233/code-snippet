import glob
from collections import defaultdict
import json
import io
import numpy as np

BEFORE = 'before-commit'
AFTER = 'after-commit'

SORT_KEY = {
    "cpu": -1,
    "before_potrf_and_magmaBatched": 0,
    "after_potrf_and_batched": 2,
}

class Markdown:
    def __init__(self):
        self.buffer = io.BufferedRandom(io.BytesIO())
        self.enc = 'utf-8'
    
    def write(self, s: str):
        self.buffer.write(s.encode(self.enc))
    
    def read(self) -> bytes:
        self.buffer.seek(0)
        return self.buffer.read()

def main():
    profs = glob.glob('./prof*.txt')

    dt_gpu = defaultdict(dict)
    dt_cpu = defaultdict(dict)
    columns = ["cpu"]

    for prof in profs:
        impl_key = prof[7:-4]
        columns.append(impl_key)

        with open(prof, 'r') as f:
            fl = f.readlines()
        
        al = [line.rstrip().split('   ') for line in fl if line.startswith('[')]

        for line in al:
            shape = line[0]
            t_cpu, t_gpu = (float(x) for x in line[-2:])

            dt_gpu[shape][impl_key] = t_gpu
            dt_cpu[shape][impl_key] = t_cpu
    
    columns.sort(key=SORT_KEY.__getitem__)
        
    print(json.dumps(dt_gpu, indent=2))
    # print(dt_cpu)

    md = Markdown()
    md.write('time is in **us** (10^-6 s)\n\n')
    md.write('|shape|' + '|'.join(columns) + '|\n')
    md.write('|---:' * (len(columns)+1) + '|\n')

    for shape in dt_gpu.keys():
        t_cpu_avg = np.mean([x for x in dt_cpu[shape].values()])
        md.write(f'| {shape} | {t_cpu_avg : .3f} |')

        for column in columns[1:]:
            md.write(f' {dt_gpu[shape].get(column, -1) : .3f} |')
        
        md.write('\n')


    with open('readme.md', 'wb') as f:
        f.write(md.read())
        

if __name__ == "__main__":
    main()