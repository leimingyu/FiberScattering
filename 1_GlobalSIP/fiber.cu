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

// texture memory
texture<float4, 1, cudaReadModeElementType> crdc_tex;
texture<float4, 1, cudaReadModeElementType> crdh_tex;
texture<float4, 1, cudaReadModeElementType> crdo_tex;

// constant memory
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

__device__ __constant__ float  q_const[2048];
__device__ __constant__ float4 formfactor_const[2048];

//----------------//
// unified memory
//----------------//
float     *q;
float4    *formfactor;
float     *Iq;
float     *Iqz;
float     *Iq_final;
float     *Iqz_final;
float4    *crd_c;
float4    *crd_h;
float4    *crd_o;

//----------------//
// parameters 
//----------------//
char     *fname;
float     lamda;
float     distance;
int       span;
int       nstreams;     // number of cuda streams
int       linenum;
int       line_c;
int       line_h;
int       line_o;

// cuda related
float elapsedTime;
cudaEvent_t start, stop;
cudaStream_t *streams;

float kernel_runtime = 0.f;

dim3 block(1, 1, 1);
dim3 grid(1, 1, 1);

std::vector<int> beginpos;
std::vector<int> endpos;

int stream_per_com;

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

void readpdb()
{
	// Steps:
	// search the first line start from ATOM
	// the 3rd column, first character is the atom type 
	// 6, 7, 8 column is the x, y, z corordinates

	char line[1000];
	char c1[20];
	char atominfo[20];
	float x, y, z;

	FILE *fp = fopen(fname,"r");
	if(fp == NULL)
		perror("Error opening file!!!\n\n");

	char s;
	linenum = 0;
	line_c  = 0;
	line_h  = 0;
	line_o  = 0;

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s", c1);
		if(!(strcmp(c1, "ATOM")))
		{
			linenum++;                               // atom list length
			sscanf(line, "%*s %*d %s", atominfo);
			s = atominfo[0];
			if(s == 'C')
				line_c++;
			if(s == 'H')
				line_h++;
			if(s == 'O')
				line_o++;
		}
	}

	rewind(fp);

	std::cout << "line number = " << linenum << std::endl;
	//std::cout << line_c << std::endl;
	//std::cout << line_h << std::endl;
	//std::cout << line_o << std::endl;

	// unified memory
	cudaMallocManaged((void**)&crd_c, sizeof(float4) * line_c);
	cudaMallocManaged((void**)&crd_h, sizeof(float4) * line_h);
	cudaMallocManaged((void**)&crd_o, sizeof(float4) * line_o);

	int id_c, id_h, id_o;
	id_c = id_h = id_o = 0;

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s", c1);
		if(!(strcmp(c1, "ATOM")))
		{
			sscanf(line, "%*s %*d %s %*s %*d %f %f %f", atominfo, &x, &y, &z);
			s = atominfo[0];
			if(s == 'C')
			{
				crd_c[id_c] = make_float4(x, y, z, 0.f);
				id_c++;
			}

			if(s == 'H')
			{
				crd_h[id_h] = make_float4(x, y, z, 0.f);
				id_h++;	
			}

			if(s == 'O')
			{
				crd_o[id_o] = make_float4(x, y, z, 0.f);
				id_o++;
			}
		}
	}

	fclose(fp);
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
		q[gid]  = local_q;

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
	}
}

// kernel_cc
__global__ void kernel_cc(int streamID,
                         int line_c,
		                 int N,
		                 int start,
		                 int end,
						 int cc_start,
		                 float* Iq,
		                 float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	int lastpos = line_c - 1;
	float iq, iqz;

	if(gid < N)
	{
		float data = q_const[gid];
		float fj_fk = powf(formfactor_const[gid].x, 2.0); // fj * fk

		// atom list iteration
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdc_tex, j); // crdc_tex

			for(int k = j + 1; k <= lastpos; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdc_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[cc_start + gid  + offset] = iq;
		Iqz[cc_start + gid + offset] = iqz;
	}
}




// kernel_hh
__global__ void kernel_hh(int streamID,
                          int line_h,
		                  int N,
		                  int start,
		                  int end,
						  int hh_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	int lastpos = line_h - 1;
	float iq, iqz;

	if(gid < N)
	{
		float data = q_const[gid];
		float fj_fk = powf(formfactor_const[gid].y, 2.0);  // y = h

		// atom list iteration
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdh_tex, j); // crdh_tex

			for(int k = j + 1; k <= lastpos; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdh_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[hh_start + gid  + offset] = iq;
		Iqz[hh_start + gid + offset] = iqz;
	}
}



// kernel_oo
__global__ void kernel_oo(int streamID,
                          int line_o,
		                  int N,
		                  int start,
		                  int end,
						  int oo_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	int lastpos = line_o - 1;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];
		float fj_fk = powf(formfactor_const[gid].z, 2.0);  // z = o

		// atom list iteration
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdo_tex, j); // crdh_tex

			for(int k = j + 1; k <= lastpos; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdo_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[oo_start  + gid + offset] = iq;
		Iqz[oo_start + gid + offset] = iqz;
	}
}



// kernel_oc: when line_o is longer
__global__ void kernel_oc(int streamID,
                          int len_c,
		                  int N,
		                  int start,
		                  int end,
						  int co_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];

		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].z * formfactor_const[gid].x;

		// iterate throught the o atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdo_tex, j);

			// compare with the c atom list
			for(int k = 0; k < len_c; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdc_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[co_start  + gid + offset] = iq;
		Iqz[co_start + gid + offset] = iqz;
	}
}


// kernel_co: when line_c is longer
__global__ void kernel_co(int streamID,
                          int len_o,
		                  int N,
		                  int start,
		                  int end,
						  int co_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];

		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].z * formfactor_const[gid].x;

		// iterate throught the c atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdc_tex, j);

			// compare with the o atom list
			for(int k = 0; k < len_o; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdo_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[co_start  + gid + offset] = iq;
		Iqz[co_start + gid + offset] = iqz;
	}
}



// kernel_hc: when line_h is longer
__global__ void kernel_hc(int streamID,
                          int len_c,
		                  int N,
		                  int start,
		                  int end,
						  int ch_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];

		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].x * formfactor_const[gid].y;

		// iterate throught the h atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdh_tex, j);

			// compare with the c atom list
			for(int k = 0; k < len_c; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdc_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[ch_start  + gid + offset] = iq;
		Iqz[ch_start + gid + offset] = iqz;
	}
}


// kernel_ch: when line_c is longer
__global__ void kernel_ch(int streamID,
                          int len_h,
		                  int N,
		                  int start,
		                  int end,
						  int ch_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];

		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].x * formfactor_const[gid].y;

		// iterate throught the c atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdc_tex, j);

			// compare with the h atom list
			for(int k = 0; k < len_h; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdh_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[ch_start  + gid + offset] = iq;
		Iqz[ch_start + gid + offset] = iqz;
	}
}



// kernel_ho: when line_h is longer
__global__ void kernel_ho(int streamID,
                          int len_o,
		                  int N,
		                  int start,
		                  int end,
						  int ho_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];
		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].y * formfactor_const[gid].z;

		// iterate throught the h atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdh_tex, j);

			// compare with the o atom list
			for(int k = 0; k < len_o; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdo_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[ho_start  + gid + offset] = iq;
		Iqz[ho_start + gid + offset] = iqz;
	}
}


// kernel_oh: when line_o is longer
__global__ void kernel_oh(int streamID,
                          int len_h,
		                  int N,
		                  int start,
		                  int end,
						  int ho_start,
		                  float* Iq,
		                  float* Iqz)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int offset = N * streamID;
	float iq, iqz;

	if(gid < N)
	{
		float data  = q_const[gid];
		// x: c		y: h	z: o
		float fj_fk = formfactor_const[gid].y * formfactor_const[gid].z;

		// iterate throught the o atom list 
		for (int j = start; j <= end; ++j) 
		{
			float4 crd_ref = tex1Dfetch(crdo_tex, j);

			// compare with the h atom list
			for(int k = 0; k < len_h; ++k)
			{
				float4 distance = crd_ref - tex1Dfetch(crdh_tex, k);

				iq  += fj_fk * j0(data * sqrt(distance.x * distance.x + distance.y * distance.y));
				iqz += fj_fk * cosf(fabsf(distance.z) * data);
			}
		}

		// accumulate the iterated results 
		Iq[ho_start  + gid + offset] = iq;
		Iqz[ho_start + gid + offset] = iqz;
	}
}




__global__ void kernel_sum(float *Iq,
                           float *Iqz,
                           int    nstreams,
                           int    N,
                           float *Iq_final,
                           float *Iqz_final)
{
	size_t gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	// nstreams for each combination  and 6 combinations in total
	int iterations = nstreams;

	if(gid < N)
	{
		float tmp_iq, tmp_iqz;
		tmp_iq = tmp_iqz = 0.f;

		for(int i = 0; i < iterations; i++)
		{
			tmp_iq  +=  Iq[i * N + gid];
			tmp_iqz += Iqz[i * N + gid];
		}
		// accumulate the iterated results 
		Iq_final[gid]  = tmp_iq;
		Iqz_final[gid] = tmp_iqz;
	}
}


void sum_pairwise()
{

#if TK
	cudaEventRecord(start, 0);
#endif

	kernel_sum <<< grid, block >>> (Iq, Iqz, nstreams, span, Iq_final, Iqz_final); 

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel sum = %f ms\n", elapsedTime);

	kernel_runtime += elapsedTime;
#endif

}

// compute workloads for cc
void work_cc(int streamid)
{
	int len = line_c;
	int sid = streamid % stream_per_com;
	int step = (len - 1) / stream_per_com;

	beginpos.push_back(sid * step);

	if(sid == (stream_per_com-1))
	{
		endpos.push_back(len - 2);
	}
	else
	{
		endpos.push_back((sid + 1) * step - 1);
	}

	//std::cout << "cc sid: "<< sid << std::endl;
}

void work_hh(int streamid)
{
	int len = line_h;
	int sid = streamid % stream_per_com;
	int step = (len - 1) / stream_per_com;

	beginpos.push_back(sid * step);

	if(sid == (stream_per_com-1))
	{
		endpos.push_back(len - 2);
	}
	else
	{
		endpos.push_back((sid + 1) * step - 1);
	}

	//std::cout << "hh sid: "<< sid << std::endl;
}

void work_oo(int streamid)
{
	int len = line_o;
	int sid = streamid % stream_per_com;
	int step = (len - 1) / stream_per_com;

	beginpos.push_back(sid * step);

	if(sid == (stream_per_com-1)){
		endpos.push_back(len - 2);
	}
	else{
		endpos.push_back((sid + 1) * step - 1);
	}
	// std::cout << "oo sid: "<< sid << std::endl;
}

void work_co(int streamid)
{
	int len, step;
	int sid = streamid % stream_per_com;

	if(line_c < line_o)
	{
		len = line_o;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	else
	{
		len = line_c;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	//std::cout << "co sid: "<< sid << std::endl;
}

void work_ch(int streamid)
{
	int len, step;
	int sid = streamid % stream_per_com;

	if(line_c < line_h)
	{
		len = line_h;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	else
	{
		len = line_c;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	//std::cout << "ch sid: "<< sid << std::endl;
}

void work_ho(int streamid)
{
	int len, step;
	int sid = streamid % stream_per_com;

	if(line_o < line_h)
	{
		len = line_h;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	else
	{
		len = line_o;		
		step = len / stream_per_com;
		beginpos.push_back(sid * step);
		if(sid == (stream_per_com-1)){
			endpos.push_back(len - 1);
		}else{
			endpos.push_back((sid + 1) * step - 1);
		}
	}
	//std::cout << "ho sid: "<< sid << std::endl;
}

void run_cc(int i)
{
	kernel_cc <<< grid, block, 0, streams[i] >>> (i, line_c, span, beginpos.at(i), endpos.at(i), 0, Iq, Iqz); 
}

void run_hh(int i)
{
	int hh_start = stream_per_com * span;
	kernel_hh <<< grid, block, 0, streams[i] >>> (i, line_h, span, beginpos.at(i), endpos.at(i), hh_start, Iq, Iqz); 
}

void run_oo(int i)
{
	int oo_start = 2 * stream_per_com * span;
	kernel_oo <<< grid, block, 0, streams[i] >>> (i, line_c, span, beginpos.at(i), endpos.at(i), oo_start, Iq, Iqz); 
}

void run_co(int i)
{
	int co_start = 3 * stream_per_com * span;

	if(line_c < line_o) {
		kernel_oc <<< grid, block, 0, streams[i] >>> (i, line_c, span, beginpos.at(i), endpos.at(i), co_start, Iq, Iqz); 
	}
	else
	{
		kernel_co <<< grid, block, 0, streams[i] >>> (i, line_o, span, beginpos.at(i), endpos.at(i), co_start, Iq, Iqz); 
	}
}

void run_ch(int i)
{
	int ch_start = 4 * stream_per_com * span;

	if(line_c < line_h) {
		kernel_hc <<< grid, block, 0, streams[i] >>> (i, line_c, span, beginpos.at(i), endpos.at(i), ch_start, Iq, Iqz); 
	}
	else
	{
		kernel_ch <<< grid, block, 0, streams[i] >>> (i, line_h, span, beginpos.at(i), endpos.at(i), ch_start, Iq, Iqz); 
	}
}


void run_ho(int i)
{
	int ho_start = 5 * stream_per_com * span;

	if(line_h < line_o) {
		kernel_oh <<< grid, block, 0, streams[i] >>> (i, line_h, span, beginpos.at(i), endpos.at(i), ho_start, Iq, Iqz); 
	}else{
		kernel_ho <<< grid, block, 0, streams[i] >>> (i, line_o, span, beginpos.at(i), endpos.at(i), ho_start, Iq, Iqz); 
	}
}


int main(int argc, char*argv[])
{
	//-----------------------------------------------------------------------//
	// Read input
	//-----------------------------------------------------------------------//
	lamda = 1.033f; 
	distance = 300.f;
	span = 2048;
	nstreams = 30;

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


	if (nstreams < 6 || (nstreams % 6  != 0))
	{
		std::cout << "nstreams should be multiples of 6 and larger than 5\n";
		exit(EXIT_FAILURE);                                                      
	}

	//-----------------------------------------------------------------------//
	// GPU  
	//-----------------------------------------------------------------------//
    //int dev = 0;
    //cudaSetDevice(dev);
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, dev);
    //printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    //std::cout << "max texture1d linear: " << deviceProp.maxTexture1DLinear << std::endl;
	// set device                                                               
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




#if TK
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif


	// unified mem 
	checkCudaErrors(cudaMallocManaged((void**)&q,          sizeof(float)  * span));
	checkCudaErrors(cudaMallocManaged((void**)&formfactor, sizeof(float4) * span));

	// for each combination launch nstreams
	checkCudaErrors(cudaMallocManaged((void**)&Iq,  sizeof(float) * span * nstreams));
	checkCudaErrors(cudaMallocManaged((void**)&Iqz, sizeof(float) * span * nstreams));

	// streams
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0 ; i < nstreams ; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}

	// configure the kernel grid size
	block.x = 256;
	grid.x  = ceil( (float) span / block.x );


#if TK
	cudaEventRecord(start, 0);
#endif

	kernel_prepare <<< grid, block >>> (q, span, formfactor, 1.f/lamda, 1.f/distance);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel prepare = %f ms\n", elapsedTime);

	kernel_runtime += elapsedTime;
#endif

	// copy q and formfactor to constant memory
	cudaMemcpyToSymbol(q_const,                   q,  sizeof(float) * span, 0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(formfactor_const, formfactor, sizeof(float4) * span, 0, cudaMemcpyDeviceToDevice);

	// fixme: one thread doing the i/o
	//        2nd thread working on the gpu prepare kernel
	readpdb();


	//-----------------------------------------------------------------------//
	// plan 1 : pair wise compuation
	//-----------------------------------------------------------------------//
	//  cache the crd and atom_type in constant and texture memory
	cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTexture(NULL, crdc_tex, crd_c, float4Desc));
	checkCudaErrors(cudaBindTexture(NULL, crdh_tex, crd_h, float4Desc));
	checkCudaErrors(cudaBindTexture(NULL, crdo_tex, crd_o, float4Desc));

	// output offsets
	//   cc : 0
	//   hh : 1 * nstreams * span
	//   oo : 2 * nstreams * span
	//   co : 3 * nstreams * span
	//   ch : 4 * nstreams * span
	//   ho : 5 * nstreams * span


	stream_per_com =  nstreams / 6; 


	//std::cout << "stream_per_com : "<< stream_per_com << std::endl;
	std::cout << "line_c : "<< line_c << std::endl;
	std::cout << "line_h : "<< line_h << std::endl;
	std::cout << "line_o : "<< line_o << std::endl;

	//------------------------------------------//
	// assign the workloads
	//------------------------------------------//
	for(int i = 0; i < nstreams; i++)
	{
		if(i< stream_per_com)
		{
			work_cc(i);
		}else if (i < 2 * stream_per_com){
			work_hh(i);
		}else if (i < 3 * stream_per_com){
			work_oo(i);
		}else if (i < 4 * stream_per_com){
			work_co(i);
		}else if (i < 5 * stream_per_com){
			work_ch(i);
		}else {
			work_ho(i);
		}
	}                                      

/*
	std::cout << "size of beginpos : " << beginpos.size() << std::endl;
//	std::cout << "size of endpos : "   << endpos.size()   << std::endl;
	for(int i=0; i<beginpos.size(); i++)
	{
		std::cout << beginpos[i] << " - "<< endpos[i] << std::endl;
	}
*/

#if TK
	cudaEventRecord(start, 0);
#endif

	// when line_h is longer
	for(int i = 0; i < nstreams; i++)
	{
		if(i< stream_per_com)
		{
			run_cc(i);	
		}
		else if (i < 2 * stream_per_com)
		{
			run_hh(i);	
		}
		else if (i < 3 * stream_per_com)
		{
			run_oo(i);	
		}
		else if (i < 4 * stream_per_com)
		{
			run_co(i);	
		}
		else if (i < 5 * stream_per_com)
		{
			run_ch(i);	
		}
		else 
		{
			run_ho(i);	
		}

	}                                      

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel pair-wise = %f ms\n", elapsedTime);

	kernel_runtime += elapsedTime;
#endif

	//-----------------------------------------------------------------------//
	// sum pair wise compuation
	//-----------------------------------------------------------------------//
	checkCudaErrors(cudaMallocManaged((void**)&Iq_final,  sizeof(float) * span));
	checkCudaErrors(cudaMallocManaged((void**)&Iqz_final, sizeof(float) * span));

	sum_pairwise();

	//std::cout << span << std::endl;

	cudaDeviceSynchronize(); 


//	for(int i = 0; i < span; i++){
//		// printf("Iq[%d] = %f\n", i, Iq_final[i]);		
//	}

	std::cout << "kernels execution time = " << kernel_runtime << " ms\n";


	//-----------------------------------------------------------------------//
	// Free Resource
	//-----------------------------------------------------------------------//
	cudaUnbindTexture(crdc_tex);	
	cudaUnbindTexture(crdh_tex);	
	cudaUnbindTexture(crdo_tex);	

	cudaFree(q);
	cudaFree(formfactor);
	cudaFree(crd_c);
	cudaFree(crd_h);
	cudaFree(crd_o);
	cudaFree(Iq);
	cudaFree(Iqz);
	cudaFree(Iq_final);
	cudaFree(Iqz_final);

	for (int i = 0 ; i < nstreams ; i++){
		cudaStreamDestroy(streams[i]);
	}
	free(streams);

	cudaDeviceReset();

	exit (EXIT_SUCCESS);
}
