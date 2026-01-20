// NVIDIA Hello World
// Built with clang (NOT nvcc) + nvidia-sdk via Buck2
//
// Build: buck2 build //src:nv_hello
// Run:   buck2 run //src:nv_hello

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
  printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    std::printf("No NVIDIA devices found.\n");
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::printf("NVIDIA Hello World (Buck2 + clang)\n");
  std::printf("  Device: %s\n", prop.name);
  std::printf("  SM: %d.%d\n", prop.major, prop.minor);
  std::printf("  Compiler: clang %d.%d.%d\n", __clang_major__, __clang_minor__,
              __clang_patchlevel__);

  // Launch kernel with 4 threads
  hello_kernel<<<1, 4>>>();
  cudaDeviceSynchronize();

  return 0;
}
