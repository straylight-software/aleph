// kernels.cuh
// CUDA kernel declarations for Python bindings
#pragma once

// Scale vector elements: data[i] *= scale
void launch_vector_scale(float *data, float scale, int n);

// SAXPY: y = a*x + y
void launch_saxpy(float *y, float a, const float *x, int n);

// Dot product (result written to device memory)
void launch_dot_product(float *result, const float *a, const float *b, int n);
