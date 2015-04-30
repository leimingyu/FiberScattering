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
//#include <helper_cuda.h>


#define FOUR_PI (4*CUDART_PI_F)
#define INV_PI  (1/CUDART_PI_F)
#define TK 1 // time kernel
#define DB 0 // debug 


// global value
char* fname;
float lamda = 1.033f; 
float distance = 300.f;
int   span = 2048;

float *q;
float4 *formfactor;

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

/*
texture<float4, 1, cudaReadModeElementType> crdTex;

__device__ __constant__ char   d_atomtype[60000];


__device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float4 operator-(const float4 &a, const float4 &b)
{
	return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w - b.w);
}

class PREPARE 
{
	public:
		void readpdb(const char *file); // read pdb file
		void read(const char *file);
		vector<char>  atom_type;
		vector<float4> crd;
};


void PREPARE::read(const char *file)
{
	string str = file;	
	unsigned found = str.find_last_of(".");
	string filetype = str.substr(found+1);
	if (!filetype.compare("pdb"))
		readpdb(file);
}


void PREPARE::readpdb(const char *file)
{
	char line[1000];
	char p1[10];
	char p2[10];
	char p3[10];// type
	char p4[10];
	char p5[10];//x
	char p6[10];//y
	char p7[10];//z
	char p8[10];
	char p9[10];
	char p10[2];//type as well

	FILE *fp = fopen(file,"r");
	if(fp == NULL)
		perror("Error opening file!!!\n\n");

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s %s %s %s %s %s %s %s %s %s", p1, p2, p3, p4, p5,
				p6, p7, p8, p9, p10);
		atom_type.push_back(p3[0]); // type
		crd.push_back(make_float4(atof(p5), atof(p6), atof(p7), 0.f));
	}

	fclose(fp);
}



//-----------------------------------------------------------------------------------------------//
// GPU Kernels
//-----------------------------------------------------------------------------------------------//

__global__ void calc_qr(float *d_q, float *d_R, int N, float inv_lamda, float inv_distance)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < N)
	{
		float tmp, tmp_q, tmp_r;

		tmp = gid * 0.0732f;	

		tmp_r = inv_lamda * sin(0.5 * atan(tmp * inv_distance));
		tmp_q = FOUR_PI * tmp_r;	
		tmp_r += tmp_r;

		d_q[gid] = tmp_q;	
		d_R[gid] = tmp_r;	
	}
}


__global__ void calc_FactorTable(float *d_q, float4 *d_factor, int N)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < N)
	{
		float tmp;
		float fc, fh, fo, fn;
		tmp = d_q[gid] * 0.25 * INV_PI;
		tmp = powf(tmp,2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(-d_atomC[4] * tmp) +
			d_atomC[1] * expf(-d_atomC[5] * tmp) +
			d_atomC[2] * expf(-d_atomC[6] * tmp) +
			d_atomC[3] * expf(-d_atomC[7] * tmp) +
			d_atomC[8];

		fh = d_atomH[0] * expf(-d_atomH[4] * tmp) +
			d_atomH[1] * expf(-d_atomH[5] * tmp) +
			d_atomH[2] * expf(-d_atomH[6] * tmp) +
			d_atomH[3] * expf(-d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(-d_atomO[4] * tmp) +
			d_atomO[1] * expf(-d_atomO[5] * tmp) +
			d_atomO[2] * expf(-d_atomO[6] * tmp) +
			d_atomO[3] * expf(-d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(-d_atomN[4] * tmp) +
			d_atomN[1] * expf(-d_atomN[5] * tmp) +
			d_atomN[2] * expf(-d_atomN[6] * tmp) +
			d_atomN[3] * expf(-d_atomN[7] * tmp) +
			d_atomN[8];

		d_factor[gid] = make_float4(fc, fh, fo, fn);
	}
}

__global__ void calc_diffraction(float *d_q,
		float4 *d_factor,
		const int start,
		const int end,
		const int lastpos,
		const int N,
		float *d_Iq,
		float *d_Iqz,
		int streamID)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	double iq, iqz;
	iq = iqz = 0;

	int offset = N * streamID;

	if(gid < N)
	{
		for (int startpos = start; startpos <= end; ++startpos) 
		{

			float4 factor = d_factor[gid];
			float q = d_q[gid];

			char t1, t2;
			float fj, fk;

			// read d_atomtype 1 time, by all N threads
			t1 = d_atomtype[startpos]; // const 

			// read d_factor 1 time 
			if (t1 == 'C')
			{
				fj = factor.x; // const
			}
			else if (t1 == 'H')
			{

				fj = factor.y;
			}
			else if (t1 == 'O')
			{

				fj = factor.z;
			}
			else
			{
				fj = factor.w;
			}


			float4 crd_ref = tex1Dfetch(crdTex, startpos);

			for(int i = startpos + 1; i <= lastpos; ++i) // atoms to compare with the base atom
			{
				// read d_atomtype i times 
				t2 = d_atomtype[i];

				// read d_factor i times
				if (t2 == 'C')
				{
					fk = factor.x;
				}
				else if (t2 == 'H')
				{

					fk = factor.y;
				}
				else if (t2 == 'O')
				{

					fk = factor.z;
				}
				else
				{
					fk = factor.w;
				}

				float fj_fk = fj * fk;

				float4 cur_crd = tex1Dfetch(crdTex, i);
				float4 distance =  crd_ref - cur_crd;


				iq  += fj_fk * j0(q * sqrt(distance.x * distance.x + distance.y * distance.y));

				// Iq_z=Iq_z+fj.*fk.*exp(1i*rz.*q);
				// For complex Z=X+i*Y, exp(Z) = exp(X)*(COS(Y)+i*SIN(Y)) 
				// here, only calculate the real part
				//iqz += fj_fk * cos(abs(distance.z) * q);
				iqz += fj_fk * cosf(fabsf(distance.z) * q);

			} // end of loop
		}

		// accumulate the value
		d_Iq[gid  + offset] = (float)iq;
		d_Iqz[gid + offset] = (float)iqz;

	}// end of if (gid < N)
}






*/


void Usage(char *argv0)
{
	const char *help_msg =	
		"\nUsage: %s [options] -f filename\n\n"
		"    -f filename      :file containing atom info\n"		
		"    -l lamda         :angstrom value                 [default=1.033]\n"
		"    -d distance      :specimen to detector distance  [default=300]\n"
		"    -s span          :sampling resolution            [default=2048]\n";
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
		//tmp = q[gid] * 0.25 * INV_PI;
		//tmp = powf(tmp,2.0);
		tmp = powf(q[gid] * 0.25 * INV_PI, 2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(-d_atomC[4] * tmp) +
			 d_atomC[1] * expf(-d_atomC[5] * tmp) +
			 d_atomC[2] * expf(-d_atomC[6] * tmp) +
			 d_atomC[3] * expf(-d_atomC[7] * tmp) +
			 d_atomC[8];

		fh = d_atomH[0] * expf(-d_atomH[4] * tmp) +
			d_atomH[1] * expf(-d_atomH[5] * tmp) +
			d_atomH[2] * expf(-d_atomH[6] * tmp) +
			d_atomH[3] * expf(-d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(-d_atomO[4] * tmp) +
			d_atomO[1] * expf(-d_atomO[5] * tmp) +
			d_atomO[2] * expf(-d_atomO[6] * tmp) +
			d_atomO[3] * expf(-d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(-d_atomN[4] * tmp) +
			d_atomN[1] * expf(-d_atomN[5] * tmp) +
			d_atomN[2] * expf(-d_atomN[6] * tmp) +
			d_atomN[3] * expf(-d_atomN[7] * tmp) +
			d_atomN[8];

		formfactor[gid] = make_float4(fc, fh, fo, fn);
	}
}



void run_qr()
{
	// fixe me : use occupancy api
	dim3 block(256, 1, 1);
	dim3 grid(ceil((float) span / block.x ), 1, 1);

	kernel_qr <<< grid, block >>> (q, span, 1.f/lamda, 1.f/distance);
	
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

	kernel_factor <<< grid, block >>> (q, span, formfactor);
	
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
	int opt;
	int fflag = 0;
	extern char   *optarg;  

	// fixme: detect wrong options
	while ( (opt=getopt(argc,argv,"f:l:d:s:"))!= EOF) 
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

	//-----------------------------------------------------------------------//
	// GPU  
	//-----------------------------------------------------------------------//
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    std::cout << "max texture1d linear: " << deviceProp.maxTexture1DLinear << std::endl;

	// um 
	cudaMallocManaged((void**)&q, sizeof(float) * span);

	// step 1
	run_qr();
	

	// step 2
	cudaMallocManaged((void**)&formfactor, sizeof(float4) * span);

	run_factor();


	// release
	cudaFree(q);

	cudaDeviceReset();

/*
	PREPARE prepare;
	prepare.read(inputfile);

	int len_crd = prepare.crd.size();
	cout << "atom volume = " << len_crd << endl;

	// cpu parameters
	float lamda = 1.033f;
	float distance = 300.0f;
	int N = 2000;

	size_t bytes_n = sizeof(float) * N;

	// select device for gpu
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    std::cout << "max texture1d linear: " << deviceProp.maxTexture1DLinear << std::endl;

	cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0 ; i < nstreams ; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	// copy atom type list to constant memory
	cudaMemcpyToSymbol(d_atomtype, &prepare.atom_type[0], sizeof(char) * len_crd, 0, cudaMemcpyHostToDevice);
	
	// copy coordinates to device and bind to texture memory
	float4 *d_crd;
	cudaMalloc((void**)&d_crd, sizeof(float4) * len_crd);
	cudaMemcpy(d_crd, &prepare.crd[0], sizeof(float4) * len_crd, cudaMemcpyHostToDevice);
	cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTexture(NULL, crdTex, d_crd, float4Desc));

	// allocate other device memory
	float *d_q, *d_R;
	cudaMalloc((void**)&d_q , bytes_n);
	cudaMalloc((void**)&d_R , bytes_n);


	//-------------------------------------------------------------------------------------------//
	// start gpu code 
	//-------------------------------------------------------------------------------------------//

#if TK
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime;
#endif

#if DB
	float *q,*R;
	q   = (float*)malloc(bytes_n);
	R   = (float*)malloc(bytes_n);
#endif

	//-------------------------------------------------------------------------------------------//
	// calculate q and R 
	//-------------------------------------------------------------------------------------------//

	// configure dimensions of the kernel 
	int block_size = 256;
	int grid_size  = (N+ block_size - 1)/block_size;

#if TK
	cudaEventRecord(start, 0);
#endif

	calc_qr <<< grid_size, block_size >>> (d_q, d_R, N, 1/lamda, 1/distance);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_qr = %f ms\n", elapsedTime);
#endif

#if DB	
	cudaMemcpy(q , d_q , bytes_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(R , d_R , bytes_n, cudaMemcpyDeviceToHost);
	printf("q=\n");
	for(int i=0; i<10; ++i)
	{
		printf("%6.5f ", q[i]);
	}
	printf("\n");

	printf("R=\n");
	for(int i=0; i<10; ++i)
	{
		printf("%f ", R[i]);
	}
	printf("\n");
#endif



	//-------------------------------------------------------------------------------------------//
	// calculate atom factors
	//-------------------------------------------------------------------------------------------//
	float4 *d_factor;
	cudaMalloc((void**)&d_factor, sizeof(float4) * N);

#if TK
	cudaEventRecord(start, 0);
#endif

	int blk_factor = 256;
	calc_FactorTable <<< (N + blk_factor - 1)/blk_factor, blk_factor >>> (d_q, d_factor, N);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_FactorTable = %f ms\n", elapsedTime);
#endif

#if DB
	float4 *factor;
	factor = (float4 *)malloc(sizeof(float4) *  N);
	cudaMemcpy(factor, d_factor , sizeof(float4) * N, cudaMemcpyDeviceToHost);
	cout << "\tC\tH\tO\tN" << endl;
	for(int i=0; i < N; i++){
		cout << i << "\t" << factor[i].x << "\t" << factor[i].y << "\t" << factor[i].z << "\t" << factor[i].w << endl;
	}

#endif


	//-------------------------------------------------------------------------------------------//
	// calculate iq and iqz
	//-------------------------------------------------------------------------------------------//

	// host mem for output
	float *h_Iq  = NULL;
	float *h_Iqz = NULL;
	checkCudaErrors(cudaMallocHost((void**)&h_Iq,  sizeof(float) * N * nstreams));
	checkCudaErrors(cudaMallocHost((void**)&h_Iqz, sizeof(float) * N * nstreams));

	// device mem for output
	float *d_Iq;
	float *d_Iqz;
	cudaMalloc((void**)&d_Iq,  sizeof(float) * N * nstreams);
	cudaMalloc((void**)&d_Iqz, sizeof(float) * N * nstreams);

	// slice the wordloads for each stream
	int atomNum = prepare.atom_type.size();
	int lastpos = atomNum - 1;

	int blk_diffraction = 256;
	int grd_diffraction = (N + blk_diffraction - 1) / blk_diffraction;

	int step = (atomNum - 1) / nstreams;
	vector<int> beginpos;
	vector<int> endpos;
	for(int i=0; i<nstreams; i++){
		beginpos.push_back(i * step);

		if(i == (nstreams-1)){
			endpos.push_back(atomNum-2);
		}else{
			endpos.push_back((i + 1) * step - 1);
		}
	}


#if TK
	cudaEventRecord(start, 0);
#endif

	for(int i=0; i<nstreams; i++)
	{
		calc_diffraction <<< grd_diffraction, blk_diffraction, 0, streams[i] >>> (d_q, 
                                                                                  d_factor, 
																				  beginpos.at(i), 
																				  endpos.at(i), 
																				  lastpos, 
																				  N, 
																				  d_Iq, 
																				  d_Iqz, 
																				  i);
	}



#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_diffraction = %f ms\n", elapsedTime);
#endif


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

	for (int i = 0 ; i < nstreams ; i++){
		cudaStreamDestroy(streams[i]);
	}
	free(streams);

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
