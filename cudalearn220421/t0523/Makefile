#src = $(wildcard *.c)
#targets = $(patsubst %.c, %, $(src))
src = $(wildcard *.cu)
targets = $(patsubst %.cu, %, $(src))


#CC = gcc
CC = nvcc
#CFLAGS = -Wall -g
CFLAGS = -run

all:$(targets)

#$(targets):%:%.c
$(targets):%:%.cu
	$(CC) $< -o $@ $(CFLAGS)

.PHONY:clean all
clean:
	-rm -rf $(targets) 

