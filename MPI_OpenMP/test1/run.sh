#!/bin/sh

#cmd="mpiexec.mpich2 -np 2 -f host_file ./test1"
#cmd="mpiexec -n 1 -host compute-2-129 ./test1 : -n 1 -host compute-2-137 ./test1"
#cmd="mpiexec -n 1 -host compute-2-129 ./test1"
cmd="mpirun -np 2 -machinefile hostlist-tcp ./test1"
$cmd
