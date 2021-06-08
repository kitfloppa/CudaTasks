#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void trapezoidalIntegral(double *sum, double h, double begin) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x_1 = begin + i * h, x_2 = begin + (i + 1) * h;
	double x_3 = x_1 * x_1, x_4 = x_2 * x_2;

    sum[i] = 0.5 * (x_2 - x_1) * (x_3 + x_4);
}

__global__ void reduce(double *res) {
	unsigned int tid = threadIdx.x;

	for (size_t k = 1; k < blockDim.x; k *= 2) {
		unsigned int index = 2 * k * tid;

		if (index < blockDim.x) res[index] += res[index + k];
		__syncthreads();
	}
}

int main() {
    const double a = 0, b = 15;
    const int n = 1000;
    double h = (b - a) / n, result = 0;
    double *mas_c;

    cudaMalloc((void**)&mas_c, n * sizeof(double));
	
	trapezoidalIntegral<<<1, n>>>(mas_c, h, a);
	reduce<<<1, n>>>(mas_c);

	cudaMemcpy(&result, mas_c, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(mas_c);
	printf("Result = %f", result);
    
    return 0;
} // nvcc trapezoidal.cu -o trapezoidal
