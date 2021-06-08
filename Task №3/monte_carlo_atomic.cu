#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>

__global__ void monte_carlo(float *res, double *mass_1, double *mass_2, int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double x = mass_1[tid];
    double y = mass_2[tid];

    if (((x * x) + (y * y)) <= 1) atomicAdd(res, 1.0);
}

int main() {
    int n = 10000000, block = 256, numblock = (n + block - 1) / block;
    float *result, res = 0;
    double *mass_1, *mass_2, pi = 0;

    cudaMalloc((void**)&result, sizeof(float));
    cudaMalloc((void**)&mass_1, n * sizeof(double));
    cudaMalloc((void**)&mass_2, n * sizeof(double));

    curandGenerator_t ran;
    curandCreateGenerator(&ran, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(ran, 1234ULL^time(NULL));
    
    curandGenerateUniformDouble(ran, mass_1, n);
    curandGenerateUniformDouble(ran, mass_2, n);

    monte_carlo<<<numblock, block>>>(result, mass_1, mass_2, n);
    cudaMemcpy(&res, result, sizeof(int), cudaMemcpyDeviceToHost);
    pi = 4 * res / n;

    cudaFree(result);
    cudaFree(mass_1);
    cudaFree(mass_2);
    printf("PI = %f", pi);

    return 0;
} // nvcc monte_carlo_atomic.cu -o monte_carlo_atomic -l curand
