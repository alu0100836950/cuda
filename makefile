default: cuda

cuda:
	nvcc prueba_cuda.cu -o cuda.run `pkg-config --libs --cflags opencv`

run:
	./cuda.run lena.jpg lena_out.jpg
