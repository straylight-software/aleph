// bindings.cpp
// pybind11 bindings for CUDA kernels
//
// Demonstrates:
//   - numpy array to device memory transfer
//   - Kernel launch from pybind11
//   - Device memory management

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// From kernels.cu
void launch_vector_scale(float* data, float scale, int n);
void launch_saxpy(float* y, float a, const float* x, int n);
void launch_dot_product(float* result, const float* a, const float* b, int n);

namespace py = pybind11;

// =============================================================================
// Scale array elements on GPU
// =============================================================================

py::array_t<float> scale_array(py::array_t<float> input, float scale) {
    py::buffer_info buf = input.request();

    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }

    int n = static_cast<int>(buf.size);
    float* ptr = static_cast<float*>(buf.ptr);

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    launch_vector_scale(d_data, scale, n);

    // Copy result back
    auto result = py::array_t<float>(n);
    py::buffer_info result_buf = result.request();
    cudaMemcpy(result_buf.ptr, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    return result;
}

// =============================================================================
// SAXPY: y = a*x + y on GPU
// =============================================================================

py::array_t<float> saxpy(py::array_t<float> y, float a, py::array_t<float> x) {
    py::buffer_info y_buf = y.request();
    py::buffer_info x_buf = x.request();

    if (y_buf.ndim != 1 || x_buf.ndim != 1) {
        throw std::runtime_error("Arrays must be 1-dimensional");
    }
    if (y_buf.size != x_buf.size) {
        throw std::runtime_error("Arrays must have same size");
    }

    int n = static_cast<int>(y_buf.size);

    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x_buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    launch_saxpy(d_y, a, d_x, n);

    auto result = py::array_t<float>(n);
    cudaMemcpy(result.request().ptr, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return result;
}

// =============================================================================
// Dot product on GPU
// =============================================================================

float dot(py::array_t<float> a, py::array_t<float> b) {
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();

    if (a_buf.ndim != 1 || b_buf.ndim != 1) {
        throw std::runtime_error("Arrays must be 1-dimensional");
    }
    if (a_buf.size != b_buf.size) {
        throw std::runtime_error("Arrays must have same size");
    }

    int n = static_cast<int>(a_buf.size);

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    cudaMemcpy(d_a, a_buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    launch_dot_product(d_result, d_a, d_b, n);

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

// =============================================================================
// Check if CUDA is available
// =============================================================================

bool cuda_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

std::string cuda_device_name() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return "No CUDA devices available";
    }
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    return std::string(props.name);
}

// =============================================================================
// Module definition
// =============================================================================

PYBIND11_MODULE(gpu_module, m) {
    m.doc() = "GPU-accelerated operations via CUDA";

    m.def("cuda_available", &cuda_available,
          "Check if CUDA is available");

    m.def("cuda_device_name", &cuda_device_name,
          "Get the name of the CUDA device");

    m.def("scale_array", &scale_array,
          "Scale array elements on GPU",
          py::arg("input"), py::arg("scale"));

    m.def("saxpy", &saxpy,
          "SAXPY: y = a*x + y on GPU",
          py::arg("y"), py::arg("a"), py::arg("x"));

    m.def("dot", &dot,
          "Dot product on GPU",
          py::arg("a"), py::arg("b"));
}
