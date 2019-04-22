#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

// user defined header files
#include "cmd_parser.h"

// atom diffraction coefficients
float atom_C[]={2.31000, 1.02000, 1.58860, 0.865000, 
		20.8439, 10.2075, 0.568700, 51.6512, 
		0.2156};

float atom_H[]={0.493002, 0.322912, 0.140191, 0.040810, 
		10.5109, 26.1257, 3.14236, 57.7997, 
		0.003038};

float atom_O[]={3.04850, 2.28680, 1.54630, 0.867000, 
		13.2771, 5.70110, 0.323900, 32.9089, 
		0.2508};


/* cpu parameters */
int N = 2000; // diffraction length
int length = 300000;
int count;
char *atom_type;
int *atom_num;  // ???
float *R, *q;
float *f_C, *f_H, *f_O;
float *crd_X, *crd_Y, *crd_Z; 
float *fjk, *Iq, *Iqz;

/* mpi parameters */
int my_rank;
int numproc;
int mystart, myend; // divide the loop for MPI
double timer;

/* functions */
void readpdb(const char *);
void readcor(const char *);
void cor2pdb(const char *);
void calcRq(float , float , int , char **);
void calcForm();
void save(const char *, const char *);
void formatrix();
void calcIn(int , char **);
void release();


int main(int argc, char *argv[]){

	/* parse commandline */
	read_command_line(&argc, argv);

	//std::cout << sourcefile << std::endl;
	//printf("sourcefile = %s\n", sourcefile);	

	/* read sourcefile */
	readpdb(sourcefile);

	
	/* diffraction vector */
	float lamda = 1.033f;	// angstrom unit
	float dectdist = 300.0f; // specimen to detector distance

	N = 2000; // diffraction range
	calcRq(1.f/lamda, 1.f/dectdist, argc, argv);
	calcForm();
	//save("output_r.txt", "k");

	/* simulation */
	formatrix();
	calcIn(argc, argv);
	//program.save("output_iq2iqz.txt", "iq2iqz");

	release();

	exit(EXIT_SUCCESS);
}

void readpdb(const char *file)
{
	atom_num  = (int*)  malloc(sizeof(int)  *  length);
	atom_type = (char*) malloc(sizeof(char) *  length);
	crd_X     = (float*)malloc(sizeof(float) * length);
	crd_Y     = (float*)malloc(sizeof(float) * length);
	crd_Z     = (float*)malloc(sizeof(float) * length);

	int i = 0;
	int n = 0;
	char line[80];
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

	FILE *fp;
	fp = fopen(file,"rb");
	if(fp == NULL)
	{
		perror("Error opening file!!!\n\n");
	}

	while (fgets(line,80,fp)!=NULL)
	{
		sscanf(line,"%s %s %s %s %s %s %s %s %s %s", p1,p2,p3,p4,p5,p6,p7,p8,p9,p10);
		atom_num[i]  = i;
		atom_type[i] = p3[0];// type
		crd_X[i] = (float)atof(p5);//x
		crd_Y[i] = (float)atof(p6);//y
		crd_Z[i] = (float)atof(p7);//z
		i++;
	}
	count = i; // # of atoms

	fclose(fp);
}

void readcor(const char *file)
{
	atom_num  = (int*)  malloc(sizeof(int)  *  length);
	atom_type = (char*) malloc(sizeof(char) *  length);
	crd_X     = (float*)malloc(sizeof(float) * length);
	crd_Y     = (float*)malloc(sizeof(float) * length);
	crd_Z     = (float*)malloc(sizeof(float) * length);

	int i,j,jj;
	int n=0;
	char line[200];
	char p1[20];
	char p2[20];
	char p3[20];
	char p4[20];
	char p5[20];//x
	char p6[20];//y
	char p7[20];//z
	char p8[20];
	char p9[20];
	char p10[20];//type

	j=0;
	jj=3;

	FILE *fp;
	fp=fopen(file,"rb");
	while (fgets(line,200,fp)!=NULL){
		sscanf(line,"%s %s %s %s %s %s %s %s %s %s", p1,p2,p3,p4,p5,p6,p7,p8,p9,p10);
		if (j>jj){
			i=j-jj-1;
			count=i; // # of atoms
			atom_num[i]=i;
			atom_type[i]=p4[0];// type
			crd_X[i]=(float)atof(p5);//x
			crd_Y[i]=(float)atof(p7);//y
			crd_Z[i]=(float)atof(p6);//z
			j++;
		}
		else {
			j++;
		}
	}
	fclose(fp);
}

void cor2pdb(const char *file)
{
	int i,j;
	FILE *fp;
	fp=fopen(file,"wb");
	for (i=0;i<=count;i++){
		j=i+1;
		fprintf(fp, "ATOM"); //"ATOM" 1-4

		if (j>0 && j<10){
			fprintf(fp, "      %d", j); //7-11 atom serial number
		}
		else if (j>=10 && j<100){
			fprintf(fp, "     %d", j); //7-11 atom serial number
		}
		else if (j>=100 && j<1000){
			fprintf(fp, "    %d", j); //7-11 atom serial number
		}
		else if (j>=1000 && j<10000){
			fprintf(fp, "   %d", j); //7-11 atom serial number
		}
		else if (j>=10000 && j<100000){
			fprintf(fp, "  %d", j); //7-11 atom serial number
		}		
		else {
			fprintf(fp, " %d", j); //7-11 atom serial number
		}

		fprintf(fp, " %c   ", atom_type[i]); //13-16 atom name
		fprintf(fp, " "); // 17 atlernative location indicator
		fprintf(fp,"   "); // residue name 18-20 
		fprintf(fp,"  "); // chain identifier 22
		fprintf(fp, "    "); // residue sequence number 23-26 
		fprintf(fp," "); // code for insertion of residues 27
		fprintf(fp,"   %8.3f", crd_X[i]); // x coordinate 31-38
		fprintf(fp,"%8.3f", crd_Y[i]); // y coordinate 39-46
		fprintf(fp,"%8.3f", crd_Z[i]); // z coordinate 47-54 
		fprintf(fp,"      "); // occupacy 55-60
		fprintf(fp,"      "); // temperature factor 61-66
		fprintf(fp,"      "); // segment identifier 73-76
		fprintf(fp,"%c  \n", atom_type[i]); // element symbol 77-78
	}
	fclose(fp);
}

void calcRq(float lamda_inv, float distan_inv, int argc, char *argv[])
{
	R = (float*) malloc(sizeof(float) * N);
	q = (float*) malloc(sizeof(float) * N);

	int i;


	//MPI_Status status;
//	MPI_Init(&argc, &argv);
//	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
//
//	/* divide loop */
//	mystart = (N / numproc) * my_rank;
//	if (N % numproc > my_rank){
//		mystart += my_rank;
//		myend = mystart + (N / numproc) + 1;
//	}else{
//		mystart += N % numproc;
//		myend = mystart + (N / numproc);
//	}
//	printf("CPU%d %d ~ %d\n", my_rank, mystart, myend);
//
//	MPI_Barrier(MPI_COMM_WORLD);
//	timer = -MPI_Wtime();

	for(i = 0; i < N; i++)
	//for(i = mystart; i < myend; i++)
	{
		R[i] = 2 * lamda_inv * sin(0.5 * atan(0.0732f * i * distan_inv));
		q[i] = 2 * M_PI * R[i]; // constant in math.h
	}

//	MPI_Barrier(MPI_COMM_WORLD);
//	timer += MPI_Wtime();
//
//	MPI_Finalize();
//
//	if (my_rank == 0) 
//		printf("elapsed time = %lf (s)\n", timer);
}

// form factor for each atoms
void calcForm()
{
	float temp;
	f_C = (float*) malloc(sizeof(float) * N);
	f_H = (float*) malloc(sizeof(float) * N);
	f_O = (float*) malloc(sizeof(float) * N);

	int i, j;
	for(i = 0; i < N; i++)
	{

		f_C[i] = atom_C[8]; // last coeff
		f_H[i] = atom_H[8];
		f_O[i] = atom_O[8];

		temp = pow(q[i] * M_PI_4, 2.f);

		for(j = 0; j < 4 ; j++)
		{
			f_C[i] = f_C[i] + atom_C[j] * exp(-atom_C[j + 4] * temp);
			f_H[i] = f_H[i] + atom_H[j] * exp(-atom_H[j + 4] * temp);
			f_O[i] = f_O[i] + atom_O[j] * exp(-atom_O[j + 4] * temp);
		}
	}
}

void save(const char *file, const char *type)
{
	int i;

	FILE *fp;
	fp = fopen(file,"wb");
	if(fp == NULL)
	{
		perror("Can't open the output file!!!");	
		fclose(fp);
	}
	else
	{

		if (!strcmp(type,"r"))
		{
			for(i = 0; i < N; i++)
			{
				fprintf(fp, "%f\n", R[i]);
			}

		}
		else if (!strcmp(type,"q") )
		{
			for(i = 0; i < N; i++)
			{
				fprintf(fp, "%f\n", q[i]);
			}
		}
		else if (!strcmp(type,"fc") )
		{
			for(i = 0; i < N; i++)
			{
				fprintf(fp, "%f\n", f_C[i]);
			}
		}
		else if (!strcmp(type,"fo") )
		{
			for(i = 0; i < N; i++)
			{
				fprintf(fp, "%f\n", f_O[i]);
			}
		}
		else if (!strcmp(type,"fh") )
		{
			for(i = 0; i < N; i++)
			{
				fprintf(fp, "%f\n", f_H[i]);
			}
		}
		else if (!strcmp(type,"iq2iqz") )
		{
			for (i=0; i < N; i++)
			{
				fprintf(fp, "%f	%f\n", Iq[i], Iqz[i]);
			}
		}
		else
		{
			printf("Invalid save type name! Use \"r\"/\"q\"/\"fc\"/\"fo\"/\"fh\"/\"iq2iqz\".\n" 
				"Remember to remove %s\n", file);	
			fclose(fp);
			exit(EXIT_FAILURE);
		}
		fclose(fp);
	}
}

void formatrix()
{
	// flat out for each atom

	int len = count;
	// N is diffraction length
	int offset;

	// row: atomNUM  col: scatter length
	fjk = (float*) malloc(sizeof(float) * len * N);

	int i, j;

	for (i = 0; i < len; i++)
	{
		offset = i * N;

		if (atom_type[i] == 'C')
		{
			for (j = 0; j < N; j++)
			{
				fjk[offset + j] = f_C[j];
			}
		}
		else if (atom_type[i] == 'H')
		{
			for (j = 0; j < N; j++)
			{
				fjk[offset + j] = f_H[j];
			}
		}
		else if (atom_type[i] == 'O')
		{
			for (j = 0; j < N; j++)
			{
				fjk[offset + j] = f_O[j];
			}
		}
		else 
		{
			for (j = 0; j < N; j++)
			{
				fjk[offset + j] = 0.f;
			}
		}
	}
}

void calcIn(int argc, char *argv[])
{
	int i, j, n;
	int len = count;
	float rxy,rz, tmp;

	Iq  = (float*) malloc(sizeof(float) * N);
	Iqz = (float*) malloc(sizeof(float) * N);

	memset(Iq,  0, sizeof(float)*N);
	memset(Iqz, 0, sizeof(float)*N);




	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	/* divide loop */
	mystart = (len / numproc) * my_rank;
	if (N % numproc > my_rank){
		mystart += my_rank;
		myend = mystart + (len / numproc) + 1;
	}else{
		mystart += len % numproc;
		myend = mystart + (len / numproc);
	}
	printf("CPU%d %d ~ %d\n", my_rank, mystart, myend);

	MPI_Barrier(MPI_COMM_WORLD);
	timer = -MPI_Wtime();

	//for (i = 0; i < len - 1; i++)
	#pragma omp parallel for num_threads(32) private(i,j,rxy, rz, n, tmp) shared(len, N)
	for (i = mystart; i < myend; i++)
	{
		for (j = i + 1; j < len; j++)
		{
			rxy = sqrt(pow((crd_X[i] - crd_X[j]),2.f) + pow(crd_Y[i] - crd_Y[j],2.f));
			rz  = fabs(crd_Z[i]-crd_Z[j]);
			for (n = 0; n < N; n++)
			{
				tmp = fjk[i*N+n] * fjk[j*N+n];
				Iq[n]  += tmp * j0(rxy * q[n]);
				Iqz[n] += tmp * cos(rz * q[n]);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	timer += MPI_Wtime();

	MPI_Finalize();

	if (my_rank == 0) 
		printf("elapsed time = %lf (s)\n", timer);
}

void release()
{
	free(atom_type);
	free(atom_num);
	free(R); 
	free(q);
	free(f_C); 
	free(f_H);
	free(f_O);
	free(crd_X);
	free(crd_Y);
	free(crd_Z);
	free(fjk);
	free(Iq);
	free(Iqz);
}
