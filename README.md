# FiberScattering

## Introduction
The simulation algorithm is designed by Zhang Yan.<br>
We developed optimized single GPU and multi-GPU solutions for this simulation.<br>
Here is a list of different versions we implemented.
<ul>
<li>MPI + OpenMP</li>
<li>MPI + GPU</li>
</ul>


## Usage
```bash
Usage: ./fiber_(cpu/omp/cuda/mpi) [options] -f filename

    -f filename      :file containing atom info
    -l lamda         :angstrom value                 [default=1.033]
    -d distance      :specimen to detector distance  [default=300]
    -s span          :sampling resolution            [default=2048]
```
