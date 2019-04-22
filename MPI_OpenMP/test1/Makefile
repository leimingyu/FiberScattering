#CC = /usr/bin/mpicc
#CC = /usr/bin/mpic++
#CC=/usr/bin/g++
#CC=/usr/bin/mpic++.mpich2
#CC=mpicc.mpich2
CC=mpicc

INC=#-I/usr/include/mpich2/

all: test1 

test1: test1.c
	$(CC) $(INC) $+ -o $@

.PHONY: clean
clean:
	rm -rf test1 
