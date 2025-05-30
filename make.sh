#!/bin/bash

# yum install centos-release-scl scl-utils-build -y
# yum install devtoolset-9-toolchain -y
# export LD_LIBRARY_PATH=/opt/rh/devtoolset-9/root/lib:/opt/rh/devtoolset-9/root/lib64:$LD_LIBRARY_PATH
# export PATH=/opt/rh/devtoolset-9/root/bin:$PATH
# export CC=/opt/rh/devtoolset-9/root/bin/gcc
# export CXX=/opt/rh/devtoolset-9/root/bin/g++

# 1. 检查系统gcc/g++
which gcc
which g++

# 2. 如果有，export环境变量
export CC=$(which gcc)
export CXX=$(which g++)

# 3. 确保PATH优先/usr/bin
export PATH=/usr/bin:$PATH
export PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH

cd ./ocr/network/backbone/assets/dcn
sh Makefile.sh


# 此处编译设置可能需要根据具体系统环境来进行设置；不同系统环境设置可能不一样；上面注释和非注释环境变量设置都可以试试