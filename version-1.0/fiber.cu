// Molecular Scattering Simulation 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <string>
#include <helper_cuda.h>
#include <math_constants.h>

#define FOUR_PI (4*CUDART_PI_F)
#define INV_PI  (1/CUDART_PI_F)
#define TK 1 // time kernel
#define DB 1 // debug 

__device__ __constant__ float d_atomC[9]={2.31000,  1.02000,  1.58860,  0.865000, 20.8439,  10.2075, 0.568700, 51.6512,  0.2156};
__device__ __constant__ float d_atomH[9]={0.493002, 0.322912, 0.140191, 0.040810, 10.5109,  26.1257, 3.14236,  57.7997,  0.003038};
__device__ __constant__ float d_atomO[9]={3.04850,  2.28680,  1.54630,  0.867000, 13.2771,  5.70110, 0.323900, 32.9089,  0.2508};
__device__ __constant__ float d_atomN[9]={12.2126,  3.13220,  2.01250,  1.16630,  0.005700, 9.89330, 28.9975,  0.582600, -11.52};


texture<float4, 1, cudaReadModeElementType> crdTex;

__device__ __constant__ char   d_atomtype[60000];


char *inputfile;
int   nstreams;
int   linenum;
// cpu parameters
float lamda = 1.033f;
float distance = 300.0f;
int N = 2048;
size_t bytes_n;


cudaEvent_t start, stop;
cudaStream_t *streams; 
float elapsedTime;

float kernel_runtime = 0.f;

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
		std::vector<char>  atom_type;
		std::vector<float4> crd;
};


void PREPARE::read(const char *file)
{
	std::string str = file;	
	unsigned found = str.find_last_of(".");
	std::string filetype = str.substr(found+1);
	if (!filetype.compare("pdb"))
		readpdb(file);
}


void PREPARE::readpdb(const char *file)
{
	// Steps:
	// search the first line start from ATOM
	// the 3rd column, first character is the atom type 
	// 6, 7, 8 column is the x, y, z corordinates

	char line[1000];
	char c1[20];
	char atominfo[20];
	float x, y, z;

	FILE *fp = fopen(file,"r");
	if(fp == NULL)
		perror("Error opening file!!!\n\n");

	linenum = 0;

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s", c1);
		if(!(strcmp(c1, "ATOM")))
		{
			linenum++;                               // atom list length
			sscanf(line, "%*s %*d %s %*s %*d %f %f %f", atominfo, &x, &y, &z);
			atom_type.push_back(atominfo[0]);
			crd.push_back(make_float4(x, y, z, 0.f));
		}
	}
	fclose(fp);
}



//---------------------------------------------------------------------------//
// GPU Kernels
//---------------------------------------------------------------------------//
__global__ void kernel_prepare(float *d_q,
							     float4 *d_factor,
								 int N,
								 float inv_lamda,
								 float inv_distance)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (gid < N)
	{
		float tmp, local_q;
		float fc, fh, fo, fn;

		local_q = FOUR_PI * inv_lamda * sin(0.5 * atan(gid * 0.0732f * inv_distance));
		d_q[gid]  = local_q;

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
	float iq, iqz;
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
		d_Iq[gid  + offset] = iq;
		d_Iqz[gid + offset] = iqz;

	}// end of if (gid < N)
}







int main(int argc, char*argv[])
{
	// read input
	if (argc == 3)
	{
		inputfile = argv[1];	
		nstreams  = atoi(argv[2]);
	}
	else
	{
		std::cout << "Please specify file name!\nUsage: ./fiber inputfile cudastreams\n" << std::endl;
		exit (EXIT_FAILURE);
	}

	PREPARE prepare;
	prepare.read(inputfile);

	std::cout << "line number = " << linenum << std::endl;

	// create cuda streams
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0 ; i < nstreams ; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	bytes_n = sizeof(float) * N;

	// select device for gpu
	cudaDeviceProp device_prop;                                                 
	int dev_id = findCudaDevice(argc, (const char **) argv);                    
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));             

	if (!device_prop.managedMemory) {                                           
		fprintf(stderr, "Unified Memory not supported on this device\n");       
		cudaDeviceReset();                                                      
		exit(EXIT_FAILURE);                                                      
	}                                                                           

	if (device_prop.computeMode == cudaComputeModeExclusive || device_prop.computeMode == cudaComputeModeProhibited)
	{                                                                           
		fprintf(stderr, "This sample requires a device in either default or process exclusive mode\n");
		cudaDeviceReset();                                                      
		exit(EXIT_FAILURE);                                                      
	}      

	//-----------------------------------------------------------------------//
	// start gpu code 
	//-----------------------------------------------------------------------//
#if TK
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

	// copy atom type list to constant memory
	cudaMemcpyToSymbol(d_atomtype, &prepare.atom_type[0], 
	                   sizeof(char) * linenum, 0, cudaMemcpyHostToDevice);
	
	// copy coordinates to device and bind to texture memory
	float4 *d_crd;
	cudaMalloc((void**)&d_crd, sizeof(float4) * linenum);

	cudaMemcpy(d_crd, &prepare.crd[0], sizeof(float4) * linenum, cudaMemcpyHostToDevice);

	cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTexture(NULL, crdTex, d_crd, float4Desc));

	// allocate other device memory
	float *d_q;
	cudaMalloc((void**)&d_q , bytes_n);

	float4 *d_factor;
	cudaMalloc((void**)&d_factor, sizeof(float4) * N);

	// configure dimensions of the kernel 
	int block_size = 256;
	int grid_size  = (N + block_size - 1)/block_size;

#if TK
	cudaEventRecord(start, 0);
#endif

	//-----------------------------------------------------------------------//
	// calculate q and atom factors
	//-----------------------------------------------------------------------//
	kernel_prepare <<< grid_size, block_size >>> (d_q, d_factor, N, 1/lamda, 1/distance);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel prepare = %f ms\n", elapsedTime);

	kernel_runtime += elapsedTime;
#endif


#if DB	
	float *q;
	q   = (float*)malloc(bytes_n);

	cudaMemcpy(q , d_q , bytes_n, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; ++i)
	{
		// printf("q[%d] = %f\n", i, q[i]);
	}

	float4 *factor;
	factor = (float4 *)malloc(sizeof(float4) *  N);

	cudaMemcpy(factor, d_factor , sizeof(float4) * N, cudaMemcpyDeviceToHost);

	//std::cout << "C\tH\tO\tN" << std::endl;
	for(int i=0; i < N; i++)
	{
		//std::cout << factor[i].x << "\t" << factor[i].y << "\t" << factor[i].z << "\t" << factor[i].w << std::endl;
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

	int step = (atomNum - 1) / nstreams;
	std::vector<int> beginpos;
	std::vector<int> endpos;
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
		calc_diffraction <<< grid_size, block_size, 0, streams[i] >>> (d_q, 
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

	kernel_runtime += elapsedTime;
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


	printf("kernels execution time = %f ms\n", kernel_runtime);


	// release resources
	cudaUnbindTexture(crdTex);	

	for (int i = 0 ; i < nstreams ; i++){
		cudaStreamDestroy(streams[i]);
	}
	free(streams);

	cudaFree(d_q);
	cudaFree(d_factor);
	cudaFree(d_crd);
	cudaFree(d_Iq);
	cudaFree(d_Iqz);

	cudaFreeHost(h_Iq);
	cudaFreeHost(h_Iqz);

#if DB
	free(q);
	free(factor);
#endif

	checkCudaErrors(cudaDeviceReset());

	exit (EXIT_SUCCESS);
}
