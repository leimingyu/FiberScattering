#!/bin/sh 
#BSUB -J JOB.125 
#BSUB -o output_file 
#BSUB -e error_file 
#BSUB -n 63 #BSUB -q ht-10g 
#BSUB -cwd /scratch/nroy/rmpi/test1

######## THIS IS A TEMPLATE FILE FOR TCP ENABLED MPI RUNS ON THE DISCOVERY CLUSTER ########

#### #BSUB -n has a value equal to the given value for the -np option ####

# prefix for next run is entered below

# file staging code is entered below #mkdir /scratch/nroy/rmpi/test1

#### Enter your working directory below - this is the string returned from issuing the command "pwd" #### 
#### IF you stage your files this is your run directory in the high speed scratch space mounted across all compute nodes #### 
#work=/scratch/nroy/rmpi/test1
#work=/home/yu.lei/molecularscattering/MPI/test1
work=/home/yu.lei/molecularscattering/MPI_OpenMP

##################################################### 
########DO NOT EDIT ANYTHING BELOW THIS LINE######### 
##################################################### 
cd $work 
tempfile1=hostlistrun 
tempfile2=hostlist-tcp 

rm $work/$tempfile2

echo $LSB_MCPU_HOSTS > $tempfile1 
declare -a hosts 
read -a hosts < ${tempfile1} 
for ((i=0; i<${#hosts[@]}; i += 2)) ; 
do
     HOST=${hosts[$i]}
     #CORE=${hosts[(($i+1))]}
     CORE=1
     echo $HOST:$CORE >> $tempfile2 
done 
##################################################### 
########DO NOT EDIT ANYTHING ABOVE THIS LINE######### 
#####################################################

###### The example below runs a R program using Rmpi and SNOW.
###### Change only the -np option giving the number of MPI processes and the executible to use with options to it
###### DO NOT CHANGE ANYTHING ELSE BELOW FOR mpirun OPTIONS
###### MAKE SURE THAT THE "#BSUB -n" is equal 1. R will spawn the number of slaves as indicated in the R script snow_test.R
###### This will be the number in #BSUB -n above less 1. In this case #BSUB -n is 63 and in the script we will request 62 slaves.
###### When using the parallel-ib queue (IB backplane on Discovery IB nodes) for non-RDMA enabled code but the regular code, the faster
###### 56Gb/s IB TCP backplane can be used. To specify this use the -netaddr option exactly as shown. Many types of parallel code benefit
###### from this and you should test if the faster backplane shows performance improvement for 64 or more cores.

#mpirun -np 1 -machinefile hostlist-tcp R --no-save < snow_test.R

#mpirun -np 31 -machinefile hostlist-tcp ./test1

# any clean up tasks and file migration code is entered below

##################################################### 
########DO NOT EDIT ANYTHING BELOW THIS LINE######### 
##################################################### 
#rm $work/$tempfile1 
#rm $work/$tempfile2 
##################################################### 
########DO NOT EDIT ANYTHING ABOVE THIS LINE######### 
#####################################################
