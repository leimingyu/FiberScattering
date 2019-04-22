fiber_omp: fiber_omp.cpp
	g++ -Werror -O2 $^ -o $@ -lm -lrt -fopenmp

.PHONY: clean
clean:
	rm -rf fiber_omp
