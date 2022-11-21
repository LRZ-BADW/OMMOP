SHELL=bash

noinst_HEADERS=common.hpp
bin_PROGRAMS=
if HAVE_OPENMP
bin_PROGRAMS+=basic$(EXEEXT)
basic_SOURCES=basic.cpp
basic_CXXFLAGS=${OPENMP_CXXFLAGS}
basic_LDADD=
endif

.PHONY: t test
t test: all
	env | grep ^OMP_ ; true
if HAVE_OPENMP
	for E in `seq 1 10`; do RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
endif

