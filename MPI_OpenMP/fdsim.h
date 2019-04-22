#ifndef FDSIM_H
#define FDSIM_H

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <mpi.h>


// atom diffraction class
class FDsim 
{
public:
	// constructor
	FDsim();

	// destructor
	~FDsim();

	// set functions
	void setN(int);

	// get functions
	int getN();	

	// display 
	void display(const char *type);

	// save results
	void save(const char *file, const char *type);

	// compute funtions
	void calcRq(float, float, int, char **);
	void calcForm();


	// simulation
	void formatrix();
	void calcIn();

private:
};


void FDsim::display(const char *type)
{

	if (!strcmp(type,"n"))
	{
		std::cout << N << std::endl;	
	}
	else if (!strcmp(type,"r"))
	{
		puts("Diffraction R:");
		for(int i = 0; i < N; i++)
			std::cout << R[i] << std::endl;	
	}
	else if (!strcmp(type,"q"))
	{
		puts("Diffraction q:");
		for(int i = 0; i < N; i++)
			std::cout << q[i] << std::endl;	
	}
	else if (!strcmp(type,"fc"))
	{
		puts("Form Factor For Atom C:");
		for(int i = 0; i < N; i++)
			std::cout << f_C[i] << std::endl;	
	}
	else if (!strcmp(type,"fo"))
	{
		puts("Form Factor For Atom O:");
		for(int i = 0; i < N; i++)
			std::cout << f_O[i] << std::endl;	
	}
	else if (!strcmp(type,"fh"))
	{
		puts("Form Factor For Atom H:");
		for(int i = 0; i < N; i++)
			std::cout << f_H[i] << std::endl;	
	}
	else
	{
		std::cerr << "Invalid display type name!"
			  << " Use \"n\"/\"r\"/\"q\"/\"fc\"/\"fo\"/\"fh\".\n";
		exit(EXIT_FAILURE);
	}
}













#endif
