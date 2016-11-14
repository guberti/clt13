AM_CFLAGS = $(COMMON_CFLAGS) $(EXTRA_CFLAGS) -I$(top_srcdir)/src
AM_LDFLAGS = $(top_builddir)/src/libclt13.la

check_PROGRAMS = test_clt test_clt_state_new

TESTS = $(check_PROGRAMS)

test_clt_SOURCES = test_clt.c
test_clt_state_new_SOURCES = test_clt_state_new.c

all: $(TESTS)
