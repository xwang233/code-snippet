e: a_normal.out a_reuse.out
	./a_normal.out
	echo ''
	./a_reuse.out

a_normal.out: main.cpp Makefile
	nvcc main.cpp -std=c++17 -lcusolver -lmagma -lcublas -lcusparse \
		-o a_normal.out

a_reuse.out: main.cpp Makefile
	nvcc main.cpp -std=c++17 -lcusolver -lmagma -lcublas -lcusparse -DREUSE_MAGMA_QUEUE \
		-o a_reuse.out