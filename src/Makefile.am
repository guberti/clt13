lib_LTLIBRARIES = libclt13.la

libclt13_la_LDFLAGS= -no-undefined -version-info 0:0:0
libclt13_la_CFLAGS = $(COMMON_CFLAGS) $(EXTRA_CFLAGS) -fopenmp
libclt13_la_LIBADD= -lgomp

libclt13_la_SOURCES = clt13.c crt_tree.c

include_HEADERS = clt13.h
