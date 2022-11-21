#!/bin/bash
set -ex
export \
	OMP_DEBUG=enabled OMP_DISPLAY_ENV=FALSE \
	OMP_NUM_TEAMS=1 OMP_NUM_THREADS=4 
./configure CXX=CC OPENMP_CXXFLAGS=-fopenmp CXXFLAGS='-O3 -pipe'
make t
