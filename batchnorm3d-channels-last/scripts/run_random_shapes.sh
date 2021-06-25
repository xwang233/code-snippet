#!/bin/bash

PYTORCH=/opt/pytorch/pytorch
WORKSPACE=/workspace
DESTINATION=/docker
TOTAL_SHAPES=100000

unset PYTORCH_VERSION
unset PYTORCH_BUILD_VERSION

# https://github.com/pytorch/pytorch/pull/59129/commits/5dbbc5f8ef7b64af5c845d2d4fbb34b3bbeeb80c
CURRENT_COMMIT='5dbbc5f8ef7b64af5c845d2d4fbb34b3bbeeb80c'

function build {
    pushd $PYTORCH;
    git submodule update --init --recursive;
    pip install -r requirements.txt;
    for i in `seq 5`; do pip uninstall torch -y; python setup.py clean; done;
    CUDA_HOME="/usr/local/cuda" CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
        NCCL_INCLUDE_DIR="/usr/include/" NCCL_LIB_DIR="/usr/lib/" USE_SYSTEM_NCCL=1 USE_OPENCV=1 \
        TORCH_CUDA_ARCH_LIST='7.0 8.0' python setup.py develop;
    popd;
}

function ccache {
    apt-get update;
    apt-get install -y cmake;
    mkdir -p ~/ccache;
    pushd ~/ccache;
    rm -rf ccache;
    git clone https://github.com/ccache/ccache.git;
    mkdir -p ccache/build;
    pushd ccache/build;
    cmake -DCMAKE_INSTALL_PREFIX=${HOME}/ccache \
         -DENABLE_TESTING=OFF -DZSTD_FROM_INTERNET=ON ..;
    make -j$(nproc) install;
    popd;
    popd;

    mkdir -p ~/ccache/lib;
    mkdir -p ~/ccache/cuda;
    ln -s ~/ccache/bin/ccache ~/ccache/lib/cc;
    ln -s ~/ccache/bin/ccache ~/ccache/lib/c++;
    ln -s ~/ccache/bin/ccache ~/ccache/lib/gcc;
    ln -s ~/ccache/bin/ccache ~/ccache/lib/g++;
    ln -s ~/ccache/bin/ccache ~/ccache/cuda/nvcc;

    ~/ccache/bin/ccache -M 25Gi;
 
    export PATH=~/ccache/lib:$PATH;
    export CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc;
    which gcc;
}

ccache

cd $PYTORCH
git remote add xw https://github.com/xwang233/pytorch
git checkout xw/ci-all/batchnorm3d-channels-last-cudnn-spatial-persistent
build

mkdir -p $WORKSPACE

cd $WORKSPACE
git clone --recursive https://github.com/xwang233/code-snippet.git
cd code-snippet/conv3d-channels-last/scripts

first_device_name=`nvidia-smi -L | cut -d '(' -f 1 | cut -d ':' -f 2 | head -n 1 | xargs | tr -s ' ' '_'`
log_file=`date +%s`-$first_device_name.txt
python -m torch.distributed.launch --nproc_per_node `nvidia-smi -L | wc -l` \
    random_shapes.py --total-shapes $TOTAL_SHAPES 2>&1 | tee $log_file

# copy this $tmpdir to save the benchmark results
cp -r $log_file $DESTINATION
