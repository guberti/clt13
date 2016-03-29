lib_LTLIBRARIES = libclt13.la

libclt13_la_LDFLAGS= -no-undefined -version-info 0:1:0

libclt13_la_LIBADD= -lgmp -lgomp

libclt13_la_SOURCES = clt13.c

include_HEADERS = clt13.h

bin_PROGRAMS = test_clt

test_clt_SOURCES = test_clt.c

test_clt_LDADD   = libclt13.la -lm
