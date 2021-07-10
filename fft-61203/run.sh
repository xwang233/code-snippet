#!/bin/bash

set -ex

PYTORCH=/opt/pytorch/pytorch
WORKSPACE=/workspace
DESTINATION=/docker
PROFILE_PYTHON_DIR=fft-61203

unset PYTORCH_VERSION
unset PYTORCH_BUILD_VERSION

MASTER_COMMIT='6bb33d93ab94bb268d7cfb600c700585720bcdde'
PR_COMMIT='eddb291bc3977241a1dbdb3fa06956fde4d3cfe5'

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
git remote add gh https://github.com/pytorch/pytorch
git fetch gh $MASTER_COMMIT
git checkout $MASTER_COMMIT
build

mkdir -p $WORKSPACE

cd $WORKSPACE
git clone --recursive https://github.com/xwang233/code-snippet.git
cd code-snippet/$PROFILE_PYTHON_DIR
rm *.json
rm -rf data*
python calculate.py

cd $PYTORCH
git fetch gh $PR_COMMIT
git checkout $PR_COMMIT
build

cd $WORKSPACE
cd code-snippet/$PROFILE_PYTHON_DIR
python calculate.py

tmpdir=`date +%s`
mkdir -p $tmpdir
cd $tmpdir
mv ../*.json .
mv ../data-* .
ln -s ../compare.py .
python compare.py | tee readme.txt
cd ..

# copy this $tmpdir to save the benchmark results
# cp -r $tmpdir $DESTINATION
cd $WORKSPACE/code-snippet
cp -r $PROFILE_PYTHON_DIR $DESTINATION