AC_PREREQ([2.69])
AC_INIT([examples],[0.0],[Michele Martone])
AC_COPYRIGHT([Copyright (c) 2022-2022, Michele Martone])
AM_INIT_AUTOMAKE
AC_CHECK_PROGS([GREP],[ggrep grep],[false],[])
AM_CONDITIONAL([HAVE_GREP],[test x"$GREP" != x"false" ])
AC_LANG(C++)
AC_PROG_CXX(icpx amdclang++ clang++ g++ c++)
AC_OPENMP
CXXFLAGS_backup="$CXXFLAGS"; # opencl flags needed to detect its header
CXXFLAGS="$OPENMP_CXXFLAGS $CXXFLAGS";
AC_CONFIG_HEADERS([config.h])
AC_CHECK_HEADERS([omp.h])
AC_CHECK_FUNC([omp_get_team_num])
if test x"$ac_cv_func_omp_get_team_num" != x"yes"; then
	date;
	dnl AC_CHECK_LIB([OpenMP],[omp_get_team_num],[LIBS+=' -lOpenCL'],[true])
fi
CXXFLAGS="$CXXFLAGS_backup"; # restore
dnl
AC_CACHE_CHECK([whether you have C++17],
        [have_cv_cxx17], AC_RUN_IFELSE([AC_LANG_SOURCE([[
#include <cassert>
int main() { assert(__cplusplus> 201600L) ; }
]])],[have_cv_cxx17=yes],[have_cv_cxx17=no],[have_cv_cxx17=no]))
if test X$have_cv_cxx17 = Xyes ; then
	AC_MSG_NOTICE([Using >= C++17.])
else
	AC_MSG_NOTICE([Using <  C++17.])
fi
dnl
AC_CACHE_CHECK([whether you have C++20],
        [have_cv_cxx20], AC_RUN_IFELSE([AC_LANG_SOURCE([[
#include <cassert>
int main() { assert(__cplusplus> 201703L) ; }
]])],[have_cv_cxx20=yes],[have_cv_cxx20=no],[have_cv_cxx20=no]))
if test X$have_cv_cxx20 = Xyes ; then
	AC_MSG_NOTICE([Using >= C++20.])
else
	AC_MSG_NOTICE([Using <  C++20.])
fi
dnl
AC_CHECK_LIB([asan],[__asan_init],[true],[true])
AM_CONDITIONAL([HAVE_ASAN],[test x"$ac_cv_lib_asan___asan_init" = x"yes" ])
AM_CONDITIONAL([HAVE_GTEST],[test x"$want_gtest" == x"yes" ])
AM_CONDITIONAL([HAVE_CPP17],[test x"$have_cv_cxx17" == x"yes" ])
AM_CONDITIONAL([HAVE_CPP20],[test x"$have_cv_cxx20" == x"yes" ])
AM_CONDITIONAL([HAVE_OPENMP],[test x"$ac_cv_header_omp_h" == x"yes" ])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
AC_MSG_NOTICE([Created a Makefile])
