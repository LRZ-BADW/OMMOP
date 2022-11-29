#!/bin/bash
set -ex
if test Makefile.am -nt Makefile -o configure.ac -nt configure ; then autoreconf ; fi
./configure CXX=g++ CXXFLAGS='-Ofast -std=c++17 -march=native -mtune=native'
make clean
make
