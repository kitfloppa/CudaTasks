#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <numeric>

__global__ void monte_carlo(double *x, double *y, double *res, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if ((x[tid] * x[tid]) + (y[tid] * y[tid]) <= 1) res[tid] = 1;
	else res[tid] = 0;
}

int main() {
    const unsigned int n = 5000000, block = 256, numblocks = (n + block - 1) / block;
    double *result = (double*)malloc(n * sizeof(double)); 
    double *x, *y, *mass_c, res = 0, pi = 0;

	cudaMalloc(&x, n * sizeof(double));
	cudaMalloc(&y, n * sizeof(double));
	cudaMalloc(&mass_c, n * sizeof(double));

    curandGenerator_t ran;
	curandCreateGenerator(&ran, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(ran, 1234ULL);

    curandGenerateUniformDouble(ran, x, n);
	curandGenerateUniformDouble(ran, y, n);

    monte_carlo<<<numblocks, block>>>(x, y, mass_c, n);
    cudaMemcpy(result, mass_c, n * sizeof(double), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < n; ++i) res += result[i];
    pi = 4 * res / n;

    cudaFree(x);
    cudaFree(y);
    cudaFree(mass_c);
    printf("Pi = %f", pi);

    return 0;
}