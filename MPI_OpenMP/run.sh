#!/bin/sh
#cmd="mpiexec.mpich2 -np 4 ./fd --source ../pdbfiles/ec.pdb"
#cmd="mpirun -np 1 ./fd --source ../pdbfiles/ec.pdb"
#cmd="mpirun -np 2 -machinefile hostlist-tcp ./fd --source ../pdbfiles/ec.pdb"
cmd="mpirun -np 64 ./fd --source ../data/gpgpu8/m1.pdb"
$cmd
