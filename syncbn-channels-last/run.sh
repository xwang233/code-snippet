#!/bin/bash

set -ex

build_torch() {
    pushd /opt/pytorch/pytorch
    git reset --hard $1
    git submodule update --init --recursive
    pip install -r requirements.txt
    for i in `seq 5`; do
        pip uninstall torch -y
        python setup.py clean
    done
    CUDA_HOME="/usr/local/cuda" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    USE_SYSTEM_NCCL=1 \
    USE_OPENCV=1 \
    TORCH_CUDA_ARCH_LIST='7.0 8.0+PTX' \
        python setup.py develop
    popd
}

build_apex() {
    pushd /opt/pytorch/apex
    git submodule update --init --recursive
    pip install -r requirements.txt
    for i in `seq 5`; do
        pip uninstall apex -y
        python setup.py clean
    done
    TORCH_CUDA_ARCH_LIST='7.0 8.0+PTX' \
        python setup.py develop --user --cpp_ext --cuda_ext
    popd
}

unset PYTORCH_BUILD_VERSION
unset PYTORCH_VERSION

BEFORE=5e2f17d77a1d4592f571349dc952ff3ec42703de
AFTER=feabdafcb14f45aaa99adfa8eb7e9ad8a4e775ab

pushd /opt/pytorch/pytorch
git remote add gh https://github.com/pytorch/pytorch || true
git fetch gh $BEFORE
git fetch gh $AFTER
popd

build_torch $BEFORE
build_apex
python bench.py --pre

build_torch $AFTER
build_apex
python bench.py
