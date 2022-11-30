SHELL=bash

noinst_HEADERS=common.hpp
bin_PROGRAMS=
if HAVE_OPENMP
bin_PROGRAMS+=basic$(EXEEXT)
basic_SOURCES=basic.cpp
basic_CXXFLAGS=${OPENMP_CXXFLAGS}
basic_LDADD=
endif

dist_doc_DATA=rebuild_lumi_batch.sh
dist_doc_DATA+=rebuild_lumi_login.sh
dist_doc_DATA+=rebuild_gcc.sh
dist_doc_DATA+=rebuild_amd_mi_100.sh

if HAVE_OPENMP
bin_PROGRAMS+=loop$(EXEEXT)
loop_SOURCES=loop.cpp
loop_CXXFLAGS=${OPENMP_CXXFLAGS}
loop_LDADD=
endif

.PHONY: t tests
t tests: all
	env | grep ^OMP_ ; true
if HAVE_OPENMP
	./loop$(EXEEXT) | grep team=0
	set -ex; for E in `seq 1 10`; do KERNEL=-1 RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 1 10`; do KERNEL=0  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 3 10`; do KERNEL=1  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 10 11`; do KERNEL=2  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 9 11`; do KERNEL=3  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 9 11`; do KERNEL=4  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
	set -ex; for E in `seq 9 12`; do KERNEL=5  RANDOMIZED=1 N=$$((2**E)) ./basic$(EXEEXT); done
endif

