# FiberScattering

## Introduction
The simulation algorithm is designed by Zhang Yan.<br>
We developed optimized single GPU and multi-GPU solutions for this simulation.<br>
Here is a list of different versions we implemented.
<ul>
<li>CPU : single-threaded</li>
<li>CPU_OpenMP : parallelize for loop</li>
<li>CUDA : GPU implementation for a single GPU system</li>
<li>MPI_CUDA : distributed GPU implementation</li>
<li>MPI_OpenMP :  distributed CPU implementation</li>
</ul>


## Usage
```bash
Usage: ./fiber_(cpu/omp/cuda/mpi) [options] -f filename

    -f filename      :file containing atom info
    -l lamda         :angstrom value                 [default=1.033]
    -d distance      :specimen to detector distance  [default=300]
    -s span          :sampling resolution            [default=2048]
```

## Reference / Citation
Please refer to it if you want to use. 

regular
```
Yu, Leiming, Yan Zhang, Xiang Gong, Nilay Roy, Lee Makowski, and David Kaeli. "High performance computing of fiber scattering simulation." In Proceedings of the 8th Workshop on General Purpose Processing using GPUs, pp. 90-98. ACM, 2015.
```

bibtex
```
@inproceedings{yu2015high,
  title={High performance computing of fiber scattering simulation},
  author={Yu, Leiming and Zhang, Yan and Gong, Xiang and Roy, Nilay and Makowski, Lee and Kaeli, David},
  booktitle={Proceedings of the 8th Workshop on General Purpose Processing using GPUs},
  pages={90--98},
  year={2015},
  organization={ACM}
}
```
