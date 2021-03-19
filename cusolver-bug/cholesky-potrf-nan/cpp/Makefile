e: a.out
	./a.out

a.out: main.cpp Makefile data.bin
	nvcc main.cpp -std=c++17 -lcusolver -lblas -llapack -D USE_LAPACK
	# nvcc main.cpp -std=c++17 -lcusolver