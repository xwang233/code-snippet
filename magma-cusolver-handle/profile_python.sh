#!/bin/bash

echo 'run python script directly, get wall time'
python a.py

echo 'run nsys profile, get nsight report file'
nsys profile \
    -t cublas,cuda,nvtx,osrt \
    --cudabacktrace all \
    -c cudaProfilerApi \
    --stop-on-range-end false \
    --osrt-backtrace-threshold 50000 \
    python a.py