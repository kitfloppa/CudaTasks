#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_ERR 1e-6

__global__ void vector_add(double *res, double *a, double *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) res[tid] = a[tid] + b[tid];
}

__global__ void reduce(double *res) {
	unsigned int tid = threadIdx.x;

	for (size_t k = blockDim.x / 2; k > 0; k >>= 1) {
		if (tid < k) res[tid] += res[tid + k];
		__syncthreads();
	}
}

int main() {
    int n = 256, grid_size = ((n + n) / n);;
    double *a, *b, *c, sum = 0;
    double *d_a, *d_b, *d_c; 

    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    c = (double*)malloc(n * sizeof(double));

    for(size_t i = 0; i < n; ++i){
        a[i] = 1010.0;
        b[i] = 1011.0;
    }

    cudaMalloc((void**)&d_a, n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMalloc((void**)&d_c, n * sizeof(double));

    cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    vector_add<<<grid_size, n>>>(d_c, d_a, d_b, n);
    cudaMemcpy(c, d_c, n * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < n; ++i) assert(fabs(c[i] - a[i] - b[i]) < MAX_ERR);

    reduce<<<grid_size, n>>>(d_c);
    cudaMemcpy(&sum, d_c, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a); 
    free(b); 
    free(c);

    printf("Sum = %f", sum);

    return 0;
} // nvcc reduce.cu -o reduce
