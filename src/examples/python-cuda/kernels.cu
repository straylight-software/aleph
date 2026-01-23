// kernels.cu
// CUDA kernels for Python bindings
//
// Compiled with clang -x cuda (NOT nvcc)

#include "kernels.cuh"
#include <cuda_runtime.h>

// =============================================================================
// Vector scale kernel
// =============================================================================

__global__ void vector_scale_kernel(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

void launch_vector_scale(float* data, float scale, int n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    vector_scale_kernel<<<num_blocks, block_size>>>(data, scale, n);
    cudaDeviceSynchronize();
}

// =============================================================================
// SAXPY kernel: y = a*x + y
// =============================================================================

__global__ void saxpy_kernel(float* y, float a, const float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

void launch_saxpy(float* y, float a, const float* x, int n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    saxpy_kernel<<<num_blocks, block_size>>>(y, a, x, n);
    cudaDeviceSynchronize();
}

// =============================================================================
// Dot product kernel (parallel reduction)
// =============================================================================

__global__ void dot_product_kernel(float* result, const float* a, const float* b, int n) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one product
    shared_data[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

void launch_dot_product(float* result, const float* a, const float* b, int n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    dot_product_kernel<<<num_blocks, block_size>>>(result, a, b, n);
    cudaDeviceSynchronize();
}
