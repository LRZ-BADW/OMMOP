SHELL=bash

bin_PROGRAMS=
if HAVE_OPENMP
bin_PROGRAMS+=basic
basic_SOURCES=basic.cpp
basic_CXXFLAGS=${OPENMP_CXXFLAGS}
basic_LDADD=
endif

test: all
if HAVE_OPENMP
	./basic$(EXEEXT)
endif

