CC 	   = gcc
CFLAGS = -Wall -Wno-unused-label --std=gnu11 -g -fopenmp
IFLAGS = -Isrc -Isrc/parser
LFLAGS = -lgmp -lm

SRCS   = src/clt13.c 
HEADS  = src/clt13.h
OBJS   = $(addsuffix .o, $(basename $(SRCS)))

all: test_mmap

test_mmap: $(OBJS) $(SRCS) test_clt.c
	$(CC) $(CFLAGS) $(IFLAGS) $(LFLAGS) $(OBJS) test_clt.c -o test_clt

src/%.o: src/%.c 
	$(CC) $(CFLAGS) $(IFLAGS) -c -o $@ $<

clean:
	$(RM) test_clt
	$(RM) $(OBJS)
	$(RM) -r *.mmap
	$(RM) -r *.pp
