SHELL=bash

noinst_HEADERS=common.hpp
bin_PROGRAMS=
if HAVE_OPENMP
bin_PROGRAMS+=basic$(EXEEXT)
basic_SOURCES=basic.cpp
basic_CXXFLAGS=${OPENMP_CXXFLAGS}
basic_LDADD=
endif

if HAVE_OPENMP
bin_PROGRAMS+=loop$(EXEEXT)
loop_SOURCES=loop.cpp
loop_CXXFLAGS=${OPENMP_CXXFLAGS}
loop_LDADD=
endif

.PHONY: t test
t test: all
	env | grep ^OMP_ ; true
if HAVE_OPENMP
	./loop$(EXEEXT) | grep team=0
	set -ex; for E in `seq 1 10`; do KERNEL=-1 RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 1 10`; do KERNEL=0  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
endif

