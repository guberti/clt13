AC_PREREQ(2.65)
AC_INIT([libclt], [0.1.0], [???])

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

CFLAGS=

AC_CHECK_HEADERS([gmp.h omp.h])

CFLAGS="$CFLAGS -Wall -Wformat -Wformat-security -Wextra -Wunused \
-Wshadow -Wmissing-prototypes -Wfloat-equal -Wpointer-arith -Wcast-align \
-Wstrict-prototypes -Wredundant-decls -Wendif-labels -Wcast-qual -march=native \
-std=gnu11 -Wpedantic -lgmp -lgomp"

AC_FUNC_MALLOC

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
