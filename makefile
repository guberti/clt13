CC 	   = gcc
CFLAGS = -Wall -Wno-unused-result -Wno-switch \
		 --std=gnu11 -g \
		 -fopenmp \
		 -DYY_NO_UNPUT=1 -DYY_NO_INPUT=1
IFLAGS = -Isrc -Isrc/parser
LFLAGS = -lgmp -lm

SRCS   = clt13.c util.c
HEADS  = clt13.h util.h
OBJS   = $(addsuffix .o, $(basename $(SRCS)))

all: test_mmap

test_mmap: $(OBJS) $(SRCS) test_clt.c
	$(CC) $(CFLAGS) $(IFLAGS) $(LFLAGS) $(OBJS) test_clt.c -o test_mmap

src/%.o: src/%.c 
	$(CC) $(CFLAGS) $(IFLAGS) -c -o $@ $<

clean:
	$(RM) test_mmap
	$(RM) $(OBJS)
