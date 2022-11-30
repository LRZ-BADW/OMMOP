#!/bin/bash
MODULESCMD='module load PrgEnv-cray; module load rocm; module load craype-accel-amd-gfx90a;'
eval "${MODULESCMD}"
set -ex
if ! test -f configure; then autoreconf ; fi
export \
	OMP_DEBUG=enabled OMP_DISPLAY_ENV=FALSE \
	OMP_NUM_TEAMS=1 OMP_NUM_THREADS=4 
./configure CXX=CC OPENMP_CXXFLAGS=-fopenmp CXXFLAGS='-O3 -pipe -std=c++17 -DCPU_DEFAULTS=0'
make clean
make
make t
