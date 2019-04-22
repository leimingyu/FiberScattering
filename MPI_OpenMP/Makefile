#CC = /usr/bin/mpicc
#CC = /usr/bin/mpic++
#CC=/usr/bin/g++
#CC=/usr/bin/mpic++.mpich2
CC=mpicc

INC=#-I/usr/include/mpich2/

all: fd  

fd: fd.c
	$(CC) $(INC) $+ -o $@ -lm -fopenmp

.PHONY: clean
clean:
	rm -rf fd 
