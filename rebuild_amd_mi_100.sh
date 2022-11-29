#!/bin/bash
set -ex
clang++ --version | grep 'AMD clang version 14'
./configure CXX=clang++ CXXFLAGS='-O3 -fopenmp -std=c++17 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -DCPU_DEFAULTS=1'
make
make tests
N=2048 OMP_NUM_TEAMS=64 OMP_NUM_THREADS=128 KERNEL=4 ./basic
