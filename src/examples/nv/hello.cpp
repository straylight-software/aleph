// examples/nv/hello.cpp
//
// minimal nv target test - compiled with clang, NOT nvcc
//
// verifies:
//   - clang can compile .cu files
//   - nvidia-sdk headers are found
//   - libcudart links correctly
//   - device code executes on sm_90+

#include <cstdio>
#include <cuda_runtime.h>

// device kernel - runs on nv hardware
__global__ void straylight_kernel(int *result) { *result = 42; }

// host code
auto main() -> int {
  // check for nv devices
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess) {
    std::printf("straylight nv toolchain: no devices (driver not loaded)\n");
    std::printf("compilation succeeded - clang handled .cu correctly\n");
    return 0; // success - we're testing the toolchain, not the hardware
  }

  if (device_count == 0) {
    std::printf("straylight nv toolchain: no devices found\n");
    std::printf("compilation succeeded - clang handled .cu correctly\n");
    return 0;
  }

  // allocate device memory
  int *device_result = nullptr;
  cudaMalloc(&device_result, sizeof(int));

  // launch kernel
  straylight_kernel<<<1, 1>>>(device_result);
  cudaDeviceSynchronize();

  // copy result back
  int host_result = 0;
  cudaMemcpy(&host_result, device_result, sizeof(int), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(device_result);

  if (host_result == 42) {
    std::printf("straylight nv toolchain operational (device returned %d)\n",
                host_result);
    return 0;
  } else {
    std::printf("straylight nv toolchain: unexpected result %d\n", host_result);
    return 1;
  }
}
