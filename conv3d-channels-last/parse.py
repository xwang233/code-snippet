import json
import glob
import io

class Markdown:
    def __init__(self):
        super().__init__()
        self.s = io.BufferedRandom(io.BytesIO())
        self.enc = 'ascii'
    
    def write(self, s: str):
        self.s.write(s.encode(self.enc))
    
    def to_file(self, fs):
        self.s.seek(0)
        if isinstance(fs, str):
            with open(fs, 'wb') as f:
                f.write(self.s.read())
        elif hasattr(fs, 'write'):
            fs.write(self.s.read())
        else:
            raise AssertionError(f'unknown fs type: {type(fs)}')


device_str = None
cudnn_ver = None

set_shapes = set()
d_cmts = {
    '89bc9a5': 'master',
    'c8cd883': 'PR'
}
d_cmts_rev = {}
for k, v in d_cmts.items():
    d_cmts_rev[v] = k

m_mfts = [
    'contiguous-forward',
    'contiguous-backward',
    'channels-last-3d-forward',
    'channels-last-3d-backward',
]
js = {}

for fn in glob.glob('./res-*.json'):
    cmt = fn[6:-5]
    assert cmt in d_cmts, f'{cmt} not in {d_cmts}'

    with open(fn, 'r') as f:
        j = json.load(f)

        if device_str is None:
            device_str = j['device']
        else:
            assert device_str == j['device'], \
                f"got different device, last {device_str}, this {j['device']}"
        
        if cudnn_ver is None:
            cudnn_ver = j['cudnn-ver']
        else:
            assert cudnn_ver == j['cudnn-ver'], \
                f"got different cudnn version, last {cudnn_ver}, this {j['cudnn-ver']}"

    
    # print(j)
    js[cmt] = j

    for shape in j:
        set_shapes.add(shape)

l_shapes = []
for shape in set_shapes:
    if not shape.startswith('('):
        continue
    l_shapes.append(shape)
l_shapes.sort()

def parse(mft: str, out_fn: str):
    assert mft in m_mfts, f'memory format to be analyzed: {mft}, is not in {m_mfts}'

    md = Markdown()

    md.write(f'### {mft}\n\n')

    md.write(f'| shape ')
    for cmt in d_cmts:
        md.write(f'| {d_cmts[cmt]} ')
    md.write('| speed improvement |\n')

    md.write('| --- ')
    for cmt in d_cmts:
        md.write(f'| --- ')
    md.write('| --- |\n')

    for shape in l_shapes:
        md.write(f'| {shape}')
        for cmt in d_cmts:
            md.write(f'| {js[cmt][shape][mft] :.3f} ')
        
        perfx = js[d_cmts_rev['master']][shape][mft] / js[d_cmts_rev['PR']][shape][mft]
        md.write(f'| {perfx :.2f}')
        md.write('|\n')

    md.write('\n' * 3)
    md.to_file(out_fn)

device_str_replace = device_str.replace(' ', '_')
with open(f'{device_str_replace}.md', 'wb') as out_f:
    out_f.write(f'''
## profiling

device is {device_str}

cudnn_ver is {cudnn_ver}

time is in ms

backward is dgrad + wgrad

shape is (n, c, dhw, oc, ks, pad, stride)

E.g. (2, 4, 128, 32, 3, 1, 1) means

```python
x = torch.randn(2, 4, 128, 128, 128, dtype=torch.half, device='cuda').requires_grad_()
x = x.to(memory_format=torch.channels_last_3d)
net = torch.nn.Conv3d(4, 32, kernel_size=3, padding=1, stride=1)
net = net.to(dtype=torch.half, device='cuda', memory_format=torch.channels_last_3d)
net(x)
```

'''.encode('ascii'))


    parse(m_mfts[2], out_f)
    parse(m_mfts[3], out_f)