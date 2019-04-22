#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // getopt
#include <time.h>
#include <sys/time.h>
#include <math.h>   // pi
#include <string.h> // strcmp
#include <iostream>
#include <vector>

#include "util_vector.h"

#define FOUR_PI (4 * M_PI)
#define INV_PI  (M_1_PI)


const float 
d_atomC[9]={ 2.31000,  1.02000,  1.58860,  0.865000, 
             20.8439,  10.2075, 0.568700,   51.6512,  
             0.2156};

const float 
d_atomH[9]={ 0.493002, 0.322912, 0.140191, 0.040810, 
             10.5109,  26.1257, 3.14236,  57.7997,
			 0.003038};

const float 
d_atomO[9]={ 3.04850,  2.28680,  1.54630,  0.867000, 
             13.2771,  5.70110, 0.323900, 32.9089,
			 0.2508};

const float 
d_atomN[9]={ 12.2126,  3.13220,  2.01250,  1.16630,  
             0.005700, 9.89330, 28.9975,  0.582600, 
			 -11.52};

//----------------//
// parameters 
//----------------//
char     *fname;
float     lamda;
float     distance;
int       span;

int       line_c;
int       line_h;
int       line_o;
int       linenum;

float4    *crd_c;
float4    *crd_h;
float4    *crd_o;
float4    *crd;

float     *q;
float4    *formfactor;
char      *atomtype;
// results
float     *Iq;
float     *Iqz;


void Usage(char *argv0);
void readpdb();
void prepare();
void compute();

//-----------//
// timer
//-----------//
struct timeval timer;

void tic(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}


void toc(struct timeval *timer, const char* app)
{
	struct timeval tv_end, tv_diff;
	gettimeofday(&tv_end, NULL);
	long int diff = (tv_end.tv_usec + 1000000 * tv_end.tv_sec) -
		(timer->tv_usec  + 1000000 * timer->tv_sec);

	tv_diff.tv_sec  = diff / 1000000;
	tv_diff.tv_usec = diff % 1000000;

	printf("Elapsed time for %s = %ld.%06ld(s)\n", app, tv_diff.tv_sec, tv_diff.tv_usec);
}


//---------------------------------------------------------------------------//
// Main Function 
//---------------------------------------------------------------------------//
int main(int argc, char*argv[])
{
	//-----------------------------------------------------------------------//
	// Read command line 
	//-----------------------------------------------------------------------//
	lamda    = 1.033f; 
	distance = 300.f;
	span     = 2048;

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

	// used in compute() for results
	Iq  = new float[span];
	Iqz = new float[span];

	//-----------------------------------------------------------------------//
	// Read inputfile 
	//-----------------------------------------------------------------------//
	readpdb();

	//-----------------------------------------------------------------------//
	// Prepare: compute q and factors 
	//-----------------------------------------------------------------------//
	tic(&timer);
	prepare();
	toc(&timer, "compute q and form factors");

	//-----------------------------------------------------------------------//
	// Compute 
	//-----------------------------------------------------------------------//
	tic(&timer);
	compute();
	toc(&timer, "compute intensity");

	//-----------------------------------------------------------------------//
	// Free resources 
	//-----------------------------------------------------------------------//
	delete [] crd_c;
	delete [] crd_h;
	delete [] crd_o;
	delete [] crd;
	delete [] q;
	delete [] formfactor;
	delete [] atomtype;
	delete [] Iq;
	delete [] Iqz;

	exit (EXIT_SUCCESS);
}


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

	crd_c = new float4[line_c];
	crd_h = new float4[line_h];
	crd_o = new float4[line_o];

	crd        = new float4[linenum];
	atomtype   = new   char[linenum];

	int id_c, id_h, id_o;
	id_c = id_h = id_o = 0;

	int id = 0;

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s", c1);
		if(!(strcmp(c1, "ATOM")))
		{
			sscanf(line, "%*s %*d %s %*s %*d %f %f %f", atominfo, &x, &y, &z);
			s = atominfo[0];

			atomtype[id] = s;
			crd[id] = float4(x, y, z, 0.f);
			id++;

			if(s == 'C')
			{
				crd_c[id_c] = float4(x, y, z, 0.f);
				//printf("%f\t%f\t%f\t%f\n", crd_c[id_c].x, crd_c[id_c].y, crd_c[id_c].z, crd_c[id_c].w);
				id_c++;
			}

			if(s == 'H')
			{
				crd_h[id_h] = float4(x, y, z, 0.f);
				//printf("%f\t%f\t%f\t%f\n", crd_h[id_h].x, crd_h[id_h].y, crd_h[id_h].z, crd_h[id_h].w);
				id_h++;	
			}

			if(s == 'O')
			{
				crd_o[id_o] = float4(x, y, z, 0.f);
				//printf("%f\t%f\t%f\t%f\n", crd_o[id_o].x, crd_o[id_o].y, crd_o[id_o].z, crd_o[id_o].w);
				id_o++;
			}
		}
	}

	fclose(fp);
}

void prepare()
{
	q          = new  float[span];	
	formfactor = new float4[span];	

	float tmp, inv_lamda, inv_distance;
	inv_lamda    = 1 / lamda;
	inv_distance = 1 / distance;

	float fc, fh, fo, fn;

	#pragma unroll
	for(int i = 0; i < span; i++)	
	{
		tmp = q[i] = FOUR_PI * inv_lamda * sin(0.5 * atan(i * 0.0732f * inv_distance));

		tmp = -pow(tmp * 0.25 * INV_PI, 2.0f);

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

		formfactor[i] = float4(fc, fh, fo, fn);
	}
}


void compute()
{
	float fj, fk;
	float data, iq, iqz;

	float4 crd_ref, dist;

	for(int i = 0 ; i < span; i++)
	{
		data = q[i];
	    iq = iqz = 0.f;	
		for(int j = 0; j < (linenum - 1); j++)		
		{
			if(atomtype[j] == 'C')	
				fj = formfactor[i].x;
			else if (atomtype[j] == 'H')
				fj = formfactor[i].y;
			else 
				fj = formfactor[i].z;

			crd_ref = crd[j];
			
			for(int k = j + 1; k < linenum ; k++)		
			{
				
				if(atomtype[k] == 'C')	
					fk = formfactor[i].x;
				else if (atomtype[k] == 'H')
					fk = formfactor[i].y;
				else 
					fk = formfactor[i].z;

				dist = crd[k] - crd_ref;

				iq  += fj * fk * j0(data * sqrt(dist.x * dist.x + dist.y * dist.y));
				iqz += fj * fk * cos(data * abs(dist.z));
			}
			
		}

		Iq[i]  = iq;
		Iqz[i] = iqz;	
	}
}
