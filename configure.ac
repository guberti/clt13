AC_PREREQ(2.65)
AC_INIT([libclt], [0.1.0], [bcarmer@gmail.com])

AC_CONFIG_HEADERS([config.h])

AC_CONFIG_SRCDIR([./])
AC_CONFIG_AUX_DIR([build/autoconf])
AC_CONFIG_MACRO_DIR([build/autoconf])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])

AM_PROG_AR

LT_INIT

AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_LIBTOOL

AC_CHECK_HEADERS([gmp.h omp.h aesrand.h])

CFLAGS="$CFLAGS -Wall -Wformat -Wformat-security -Wextra -Wunused \
-Wshadow -Wmissing-prototypes -Wfloat-equal -Wpointer-arith -Wcast-align \
-Wstrict-prototypes -Wredundant-decls -Wendif-labels -Wcast-qual -march=native \
-std=gnu11 -Wpedantic -lgmp -lgomp -fopenmp"

AC_FUNC_MALLOC

AC_CONFIG_FILES([Makefile src/Makefile test/Makefile src/clt13.h])

AC_ARG_ENABLE(crt-tree, 
AS_HELP_STRING([--disable-crt-tree], [disable the CRT tree optimization]),
[if test "$enableval" = no; then
    AC_SUBST(DISABLE_CRT_TREE, ["#define DISABLE_CRT_TREE 1"])
 fi]
 )

AC_ARG_ENABLE(parallel-encode,
AS_HELP_STRING([--enable-parallel-encode], [parallelize the encode function]),
[if test "$enableval" = yes; then
    AC_SUBST(PARALLEL_ENCODE, ["#define PARALLEL_ENCODE 1"])
 fi]
 )


AC_OUTPUT
