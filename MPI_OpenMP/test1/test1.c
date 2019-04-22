//#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

// user defined header files

int main(int argc, char *argv[])
{
	int my_rank;
	int p;
//	int dest;
//	int tag = 0;
//	int source;
//	MPI_Status status;
	
	//char message[100];

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	MPI_Comm_size(MPI_COMM_WORLD, &p);

	MPI_Get_processor_name(processor_name, &name_len);

	//printf( "Hello world from process %d of %d\n", my_rank, p);
	printf( "Hello world from processor %s , rank %d out of %d processors\n", processor_name, my_rank, p);

/*
	if (my_rank != 0)
	{
		sprintf(message, "greetings from  process %d!", my_rank);
		dest = 0;

		MPI_Send(message, strlen(message+1), MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else
	{
		for(source = 1; source<p; source++)
		{
			MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}
*/


	MPI_Finalize();

	return 0;
}

