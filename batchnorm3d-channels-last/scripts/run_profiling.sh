#!/bin/bash

PYTORCH=/opt/pytorch/pytorch
WORKSPACE=/workspace
DESTINATION=/docker
PROFILE_PYTHON_DIR=batchnorm3d-channels-last

unset PYTORCH_VERSION
unset PYTORCH_BUILD_VERSION

CURRENT_COMMIT='7317dba25c9b0cee7e2b46c0b32587107265e1c5'

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
git remote add gh https://github.com/xwang233/pytorch
git fetch gh $CURRENT_COMMIT
git checkout $CURRENT_COMMIT
build

mkdir -p $WORKSPACE

cd $WORKSPACE
git clone --recursive https://github.com/xwang233/code-snippet.git
cd code-snippet/$PROFILE_PYTHON_DIR
python s.py

cd $PYTORCH
git checkout HEAD~1
build

cd $WORKSPACE
cd code-snippet/$PROFILE_PYTHON_DIR
python s.py

tmpdir=`date +%s`
mkdir -p $tmpdir
cd $tmpdir
mv ../res* .
ln -s ../parse.py .
python parse.py
ln -s *.md readme.md
cd ..

# copy this $tmpdir to save the benchmark results
# cp -r $tmpdir $DESTINATION
cd ..
cp -r $PROFILE_PYTHON_DIR $DESTINATION
