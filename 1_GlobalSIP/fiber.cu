/*
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // getopt
#include <time.h>

#include <iostream>
#include <vector>
#include <string>
#include <math_constants.h>

#include <cuda_runtime.h>
#include <helper_math.h> // float4
#include <helper_cuda.h> // check error


#define FOUR_PI (4*CUDART_PI_F)
#define INV_PI  (1/CUDART_PI_F)
#define TK 1 // time kernel
#define DB 0 // debug 


__device__ __constant__ float 
d_atomC[9]={ 2.31000,  1.02000,  1.58860,  0.865000, 
             20.8439,  10.2075, 0.568700,   51.6512,  
             0.2156};

__device__ __constant__ float 
d_atomH[9]={ 0.493002, 0.322912, 0.140191, 0.040810, 
             10.5109,  26.1257, 3.14236,  57.7997,
			 0.003038};

__device__ __constant__ float 
d_atomO[9]={ 3.04850,  2.28680,  1.54630,  0.867000, 
             13.2771,  5.70110, 0.323900, 32.9089,
			 0.2508};

__device__ __constant__ float 
d_atomN[9]={ 12.2126,  3.13220,  2.01250,  1.16630,  
             0.005700, 9.89330, 28.9975,  0.582600, 
			 -11.52};


// global value
char* fname;
float lamda;
float distance;
int   span;
int   nstreams;

float *q;
float4 *formfactor;

// event timing
cudaEvent_t start, stop;
float elapsedTime;

// cuda streams
cudaStream_t *streams;


void Usage(char *argv0)
{
	const char *help_msg =	
		"\nUsage: %s [options] -f filename\n\n"
		"    -f filename      :file containing atom info\n"		
		"    -l lamda         :angstrom value                 [default=1.033]\n"
		"    -d distance      :specimen to detector distance  [default=300]\n"
		"    -s span          :sampling resolution            [default=2048]\n"
		"    -n nstreams      :number of cuda streams         [default=2]\n";
	fprintf(stderr, help_msg, argv0);
	exit(-1);
}


__global__ void kernel_qr(float *q, 
                          int N, 
						  float inv_lamda, 
						  float inv_distance)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (gid < N)
	{
		q[gid] = FOUR_PI * inv_lamda * sin(0.5 * atan( gid * 0.0732f * inv_distance));
	}
}

__global__ void kernel_prepare(float *q,
                                 int N,
							     float4 *formfactor,
								 float inv_lamda,
								 float inv_distance)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (gid < N)
	{
		float tmp, local_q;
		float fc, fh, fo, fn;

		local_q = FOUR_PI * inv_lamda * sin(0.5 * atan(gid * 0.0732f * inv_distance));
		tmp = -powf(local_q * 0.25 * INV_PI, 2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(d_atomC[4] * tmp) +
			 d_atomC[1] * expf(d_atomC[5] * tmp) +
			 d_atomC[2] * expf(d_atomC[6] * tmp) +
			 d_atomC[3] * expf(d_atomC[7] * tmp) +
			 d_atomC[8];

		fh = d_atomH[0] * expf(d_atomH[4] * tmp) +
			d_atomH[1] * expf(d_atomH[5] * tmp) +
			d_atomH[2] * expf(d_atomH[6] * tmp) +
			d_atomH[3] * expf(d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(d_atomO[4] * tmp) +
			d_atomO[1] * expf(d_atomO[5] * tmp) +
			d_atomO[2] * expf(d_atomO[6] * tmp) +
			d_atomO[3] * expf(d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(d_atomN[4] * tmp) +
			d_atomN[1] * expf(d_atomN[5] * tmp) +
			d_atomN[2] * expf(d_atomN[6] * tmp) +
			d_atomN[3] * expf(d_atomN[7] * tmp) +
			d_atomN[8];

		formfactor[gid] = make_float4(fc, fh, fo, fn);
		q[gid]          = local_q;
	}
}

__global__ void kernel_qr_factor(float *q,
                                 int N,
							     float4 *formfactor,
								 float inv_lamda,
								 float inv_distance,
								 int offset)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x + offset;

	if (gid < N)
	{
		float tmp, local_q;
		float fc, fh, fo, fn;

		local_q = FOUR_PI * inv_lamda * sin(0.5 * atan(gid * 0.0732f * inv_distance));
		tmp = -powf(local_q * 0.25 * INV_PI, 2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(d_atomC[4] * tmp) +
			 d_atomC[1] * expf(d_atomC[5] * tmp) +
			 d_atomC[2] * expf(d_atomC[6] * tmp) +
			 d_atomC[3] * expf(d_atomC[7] * tmp) +
			 d_atomC[8];

		fh = d_atomH[0] * expf(d_atomH[4] * tmp) +
			d_atomH[1] * expf(d_atomH[5] * tmp) +
			d_atomH[2] * expf(d_atomH[6] * tmp) +
			d_atomH[3] * expf(d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(d_atomO[4] * tmp) +
			d_atomO[1] * expf(d_atomO[5] * tmp) +
			d_atomO[2] * expf(d_atomO[6] * tmp) +
			d_atomO[3] * expf(d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(d_atomN[4] * tmp) +
			d_atomN[1] * expf(d_atomN[5] * tmp) +
			d_atomN[2] * expf(d_atomN[6] * tmp) +
			d_atomN[3] * expf(d_atomN[7] * tmp) +
			d_atomN[8];

		formfactor[gid + offset] = make_float4(fc, fh, fo, fn);
		q[gid + offset]          = local_q;
	}
}

__global__ void kernel_factor_v1(int N,
							     float4 *formfactor,
								 float inv_lamda,
								 float inv_distance)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (gid < N)
	{
		float tmp;
		float fc, fh, fo, fn;

		tmp = FOUR_PI * inv_lamda * sin(0.5 * atan( gid * 0.0732f * inv_distance));
		tmp = -powf(tmp * 0.25 * INV_PI, 2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(d_atomC[4] * tmp) +
			 d_atomC[1] * expf(d_atomC[5] * tmp) +
			 d_atomC[2] * expf(d_atomC[6] * tmp) +
			 d_atomC[3] * expf(d_atomC[7] * tmp) +
			 d_atomC[8];

		fh = d_atomH[0] * expf(d_atomH[4] * tmp) +
			d_atomH[1] * expf(d_atomH[5] * tmp) +
			d_atomH[2] * expf(d_atomH[6] * tmp) +
			d_atomH[3] * expf(d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(d_atomO[4] * tmp) +
			d_atomO[1] * expf(d_atomO[5] * tmp) +
			d_atomO[2] * expf(d_atomO[6] * tmp) +
			d_atomO[3] * expf(d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(d_atomN[4] * tmp) +
			d_atomN[1] * expf(d_atomN[5] * tmp) +
			d_atomN[2] * expf(d_atomN[6] * tmp) +
			d_atomN[3] * expf(d_atomN[7] * tmp) +
			d_atomN[8];

		formfactor[gid] = make_float4(fc, fh, fo, fn);
	}
}

__global__ void kernel_factor(float *q, 
                              int N,
							  float4 *formfactor)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	// fixme : use vector instruction
	if (gid < N)
	{
		float tmp;
		float fc, fh, fo, fn;
		tmp = -powf(q[gid] * 0.25 * INV_PI, 2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(d_atomC[4] * tmp) +
			 d_atomC[1] * expf(d_atomC[5] * tmp) +
			 d_atomC[2] * expf(d_atomC[6] * tmp) +
			 d_atomC[3] * expf(d_atomC[7] * tmp) +
			 d_atomC[8];

		fh = d_atomH[0] * expf(d_atomH[4] * tmp) +
			d_atomH[1] * expf(d_atomH[5] * tmp) +
			d_atomH[2] * expf(d_atomH[6] * tmp) +
			d_atomH[3] * expf(d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(d_atomO[4] * tmp) +
			d_atomO[1] * expf(d_atomO[5] * tmp) +
			d_atomO[2] * expf(d_atomO[6] * tmp) +
			d_atomO[3] * expf(d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(d_atomN[4] * tmp) +
			d_atomN[1] * expf(d_atomN[5] * tmp) +
			d_atomN[2] * expf(d_atomN[6] * tmp) +
			d_atomN[3] * expf(d_atomN[7] * tmp) +
			d_atomN[8];

		formfactor[gid] = make_float4(fc, fh, fo, fn);
	}
}

void run_qr()
{
	// fixe me : use occupancy api
	dim3 block(256, 1, 1);
	dim3 grid(ceil((float) span / block.x ), 1, 1);

//#if TK
//	cudaEventRecord(start, 0);
//#endif


	kernel_qr <<< grid, block >>> (q, span, 1.f/lamda, 1.f/distance);

//#if TK
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsedTime, start, stop);
//	printf("kernel_qr = %f ms\n", elapsedTime);
//#endif

	
#if DB
	cudaDeviceSynchronize();	

	for(int i=0; i<span; i++)
		printf("q[%d] : %f\n", i, q[i]);
#endif
}



void run_factor()
{
	// fixe me : use occupancy api
	dim3 block(256, 1, 1);
	dim3 grid(ceil((float) span / block.x ), 1, 1);

//#if TK
//	cudaEventRecord(start, 0);
//#endif

	kernel_factor <<< grid, block >>> (q, span, formfactor);

//#if TK
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsedTime, start, stop);
//	printf("kernel_factor = %f ms\n", elapsedTime);
//#endif

	
#if DB
	cudaDeviceSynchronize();	

	printf("\t\tC\t\tH\t\tO\t\tN\n");

	for(int i = 0; i < span; i++)
	{
		printf("factor[%d] :\t%f\t%f\t%f\t%f\n", i, 
		                                         formfactor[i].x,
		                                         formfactor[i].y,
		                                         formfactor[i].z,
		                                         formfactor[i].w);
	}
#endif



}


int main(int argc, char*argv[])
{
	//-----------------------------------------------------------------------//
	// Read input
	//-----------------------------------------------------------------------//
	lamda = 1.033f; 
	distance = 300.f;
	span = 2048;
	nstreams = 2;

	int opt;
	int fflag = 0;
	extern char   *optarg;  

	// fixme: detect wrong options
	while ( (opt=getopt(argc,argv,"f:l:d:s:n:"))!= EOF) 
	{                    
		switch (opt) {                                                          
			case 'f': 
			         fflag = 1;               // mandatory
			         fname = optarg;                                          
					 break;
			case 'l':
			         lamda = atof(optarg); 
					 break;                                                      
			case 'd': 
			         distance = atof(optarg);
					 break;                                                      
			case 's': 
			         span = atoi(optarg);                             
					 break;                                                      
			case 'n': 
			         nstreams = atoi(optarg);                             
					 break;                                                      
			case '?': 
			         Usage(argv[0]);                                           
					 break;                                                      
			default: 
			         Usage(argv[0]);                                            
					 break;                                                      
		}                                                                       
	}                                                                       

	if (fname == 0) Usage(argv[0]);  

	if (fflag == 0) 
	{
		fprintf(stderr, "%s: missing -f option\n", argv[0]);
        Usage(argv[0]);                                            
	} 

	// check
	std::cout << "file name : " << fname     << std::endl;
	std::cout << "lamda : "     << lamda     << std::endl;
	std::cout << "distance: "   << distance  << std::endl;
	std::cout << "span: "       << span      << std::endl;
	std::cout << "streams: "    << nstreams  << std::endl;

	//-----------------------------------------------------------------------//
	// GPU  
	//-----------------------------------------------------------------------//
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    std::cout << "max texture1d linear: " << deviceProp.maxTexture1DLinear << std::endl;

#if TK
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif




	// um 
	cudaMallocManaged((void**)&q,          sizeof(float) * span);
	cudaMallocManaged((void**)&formfactor, sizeof(float4) * span);

	// streams
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0 ; i < nstreams ; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	// step 1
	// run_qr();

	// step 2
	// run_factor();


#if TK
	cudaEventRecord(start, 0);
#endif

	kernel_prepare <<< grid, block >>> (q, span, formfactor, 1.f/lamda, 1.f/distance);


#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("runtime = %f ms\n", elapsedTime);
#endif


	// release
	cudaFree(q);
	cudaFree(formfactor);

	for (int i = 0 ; i < nstreams ; i++){
		cudaStreamDestroy(streams[i]);
	}
	free(streams);

	cudaDeviceReset();



/*

	for(int i=0; i<nstreams; i++)
	{
		checkCudaErrors(cudaMemcpyAsync(&h_Iq[i * N],   &d_Iq[i * N],  sizeof(float) * N, cudaMemcpyDeviceToHost, streams[i]));
		checkCudaErrors(cudaMemcpyAsync(&h_Iqz[i * N], &d_Iqz[i * N],  sizeof(float) * N, cudaMemcpyDeviceToHost, streams[i]));
	
	}

	cudaDeviceSynchronize();

	for(int i=0; i < N; i++){
		for(int s=1; s<nstreams; s++){
			h_Iq[i]  += h_Iq[i + s * N];	
			h_Iqz[i] += h_Iqz[i + s * N];	
		}
	}


	// release resources

	cudaUnbindTexture(crdTex);	


	cudaFree(d_q);
	cudaFree(d_R);
	cudaFree(d_factor);
	cudaFree(d_Iq);
	cudaFree(d_Iqz);
	cudaFree(d_crd);

	cudaFreeHost(h_Iq);
	cudaFreeHost(h_Iqz);

#if DB
	free(q);
	free(R);
	free(factor);
#endif

	checkCudaErrors(cudaDeviceReset());
*/
	exit (EXIT_SUCCESS);
}
