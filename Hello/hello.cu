#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello() {
    printf("Hello CUDA from GPU!!!\n");
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Hello CPU\n");
    printf("Device name: %s", prop.name);
    
    return 0;
} // nvcc -arch=sm_35 hello.cu -o hello
