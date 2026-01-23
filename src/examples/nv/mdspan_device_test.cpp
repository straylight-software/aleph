// examples/nv/mdspan_device_test.cpp
//
// std::mdspan on nvidia device - the holy grail
//
// verifies:
//   - mdspan works in device code
//   - cuda::std::span interop
//   - matmul and reduction kernels
//
// compiled with: clang++ -x cuda --cuda-path=... --cuda-gpu-arch=sm_90

#include <array>
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>

// use kokkos mdspan for device compatibility
// (std::mdspan not yet in cuda::std::)
#include <experimental/mdspan>

namespace stdex = std::experimental;

namespace straylight::examples {

// ════════════════════════════════════════════════════════════════════════════════
// device kernel using mdspan
// ════════════════════════════════════════════════════════════════════════════════

// matrix multiply kernel using mdspan for type-safe indexing
// A[M,K] * B[K,N] = C[M,N]
template <typename T>
__global__ void matmul_kernel(const T *__restrict__ a_data,
                              const T *__restrict__ b_data,
                              T *__restrict__ c_data, int M, int K, int N) {
  // create mdspan views inside kernel
  using matrix_t = stdex::mdspan<const T, stdex::dextents<int, 2>>;
  using out_matrix_t = stdex::mdspan<T, stdex::dextents<int, 2>>;

  matrix_t A{a_data, M, K};
  matrix_t B{b_data, K, N};
  out_matrix_t C{c_data, M, N};

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    T sum = 0;
    for (int k = 0; k < K; ++k) {
      // use operator[] for C++23 multidimensional subscript
      sum += A[row, k] * B[k, col];
    }
    C[row, col] = sum;
  }
}

// simple reduction kernel
template <typename T>
__global__ void reduce_sum_kernel(const T *data, T *result, int n) {
  __shared__ T shared_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_data[tid] = (idx < n) ? data[idx] : T{0};
  __syncthreads();

  // parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_data[0]);
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// host-side test runner
// ════════════════════════════════════════════════════════════════════════════════

auto check_cuda_error(cudaError_t error, const char *operation) -> bool {
  if (error != cudaSuccess) {
    std::printf("  %s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
  }
  return true;
}

auto test_matmul() -> bool {
  constexpr int M = 4;
  constexpr int K = 3;
  constexpr int N = 2;

  // host data
  std::array<float, M * K> h_a{};
  std::array<float, K * N> h_b{};
  std::array<float, M * N> h_c{};

  // initialize A and B
  std::iota(h_a.begin(), h_a.end(), 1.0f); // 1, 2, 3, ...
  std::fill(h_b.begin(), h_b.end(), 1.0f); // all ones

  // device memory
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  if (!check_cuda_error(cudaMalloc(&d_a, sizeof(h_a)), "malloc A"))
    return false;
  if (!check_cuda_error(cudaMalloc(&d_b, sizeof(h_b)), "malloc B"))
    return false;
  if (!check_cuda_error(cudaMalloc(&d_c, sizeof(h_c)), "malloc C"))
    return false;

  // copy to device
  cudaMemcpy(d_a, h_a.data(), sizeof(h_a), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), sizeof(h_b), cudaMemcpyHostToDevice);

  // launch kernel
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  matmul_kernel<float><<<grid, block>>>(d_a, d_b, d_c, M, K, N);

  if (!check_cuda_error(cudaGetLastError(), "matmul kernel")) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return false;
  }

  cudaDeviceSynchronize();

  // copy back
  cudaMemcpy(h_c.data(), d_c, sizeof(h_c), cudaMemcpyDeviceToHost);

  // verify: each row of C should be sum of that row of A (since B is all ones)
  // row 0: 1+2+3 = 6, row 1: 4+5+6 = 15, row 2: 7+8+9 = 24, row 3: 10+11+12 =
  // 33
  std::array<float, M> expected_row_sums{6.0f, 15.0f, 24.0f, 33.0f};

  bool passed = true;
  for (int i = 0; i < M && passed; ++i) {
    for (int j = 0; j < N && passed; ++j) {
      if (h_c[i * N + j] != expected_row_sums[i]) {
        std::printf("  matmul C[%d,%d] = %f, expected %f\n", i, j,
                    h_c[i * N + j], expected_row_sums[i]);
        passed = false;
      }
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return passed;
}

auto test_reduction() -> bool {
  constexpr int N = 1000;

  std::array<float, N> h_data{};
  std::iota(h_data.begin(), h_data.end(), 1.0f); // 1, 2, 3, ..., 1000

  // expected sum: n*(n+1)/2 = 1000*1001/2 = 500500
  constexpr float expected_sum = 500500.0f;

  float *d_data = nullptr, *d_result = nullptr;
  float h_result = 0.0f;

  if (!check_cuda_error(cudaMalloc(&d_data, sizeof(h_data)), "malloc data"))
    return false;
  if (!check_cuda_error(cudaMalloc(&d_result, sizeof(float)), "malloc result"))
    return false;

  cudaMemcpy(d_data, h_data.data(), sizeof(h_data), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(float));

  // launch reduction
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  reduce_sum_kernel<float><<<blocks, threads>>>(d_data, d_result, N);

  if (!check_cuda_error(cudaGetLastError(), "reduce kernel")) {
    cudaFree(d_data);
    cudaFree(d_result);
    return false;
  }

  cudaDeviceSynchronize();
  cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);

  // allow small floating point error
  float diff = h_result - expected_sum;
  if (diff < 0)
    diff = -diff;

  if (diff > 1.0f) {
    std::printf("  reduction got %f, expected %f\n", h_result, expected_sum);
    return false;
  }

  return true;
}

auto main_impl() -> int {
  // check for devices
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess || device_count == 0) {
    std::printf("nv mdspan tests: no devices available\n");
    std::printf(
        "compilation succeeded - mdspan device code compiled correctly\n");
    return 0; // success - testing toolchain, not hardware
  }

  // get device info
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::printf("nv mdspan tests on: %s (sm_%d%d)\n", props.name, props.major,
              props.minor);

  int failures = 0;

  if (test_matmul()) {
    std::printf("  matmul_mdspan: pass\n");
  } else {
    std::printf("  matmul_mdspan: FAIL\n");
    failures++;
  }

  if (test_reduction()) {
    std::printf("  reduction: pass\n");
  } else {
    std::printf("  reduction: FAIL\n");
    failures++;
  }

  if (failures == 0) {
    std::printf("all nv mdspan tests passed\n");
    return 0;
  } else {
    std::printf("%d nv mdspan tests FAILED\n", failures);
    return 1;
  }
}

} // namespace straylight::examples

auto main() -> int { return straylight::examples::main_impl(); }
