import tempfile
import subprocess
import torch

def nvprof_parser_test():
    PYTHON = 'python3'
    NVPROF = 'nvprof'

    fp_python = tempfile.NamedTemporaryFile()
    fp_python.write("""
import torch
with torch.cuda.profiler.profile():
    with torch.autograd.profiler.emit_nvtx():
        x = torch.randn(4, dtype=torch.float, device='cuda')
        net = torch.nn.ReLU().cuda()
        y = net(x).sum()""".encode('ascii'))
    fp_python.flush()

    fp_nvprof = tempfile.NamedTemporaryFile()

    subprocess.run([NVPROF, '-o', fp_nvprof.name, '-f', '--', PYTHON, fp_python.name], check=True)

    prof = torch.autograd.profiler.parse_nvprof_trace(fp_nvprof.name)

    assert isinstance(prof, list)
    assert len(prof) == 2
    assert prof[0].name.startswith('relu')
    assert prof[1].name.startswith('sum')

    fp_nvprof.close()
    fp_python.close()

if __name__ == "__main__":
    nvprof_parser_test()
