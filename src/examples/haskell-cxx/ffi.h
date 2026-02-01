// ffi.h
// C interface for Haskell FFI
//
// This header defines the C ABI that Haskell's FFI can call.
// The implementation is in C++ (ffi.cpp) but uses extern "C" linkage.

#ifndef ALEPH_FFI_H
#define ALEPH_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Simple arithmetic (demonstrates basic FFI)
// =============================================================================

int32_t ffi_add(int32_t a, int32_t b);
int32_t ffi_multiply(int32_t a, int32_t b);

// =============================================================================
// Vector operations (demonstrates pointer passing)
// =============================================================================

// Compute dot product of two vectors
// Returns the result, vectors are not modified
double ffi_dot_product(const double *a, const double *b, size_t len);

// Compute L2 norm of a vector
double ffi_norm(const double *v, size_t len);

// Scale a vector in-place: v[i] *= scalar
void ffi_scale(double *v, size_t len, double scalar);

// =============================================================================
// String operations (demonstrates string passing)
// =============================================================================

// Returns a greeting string. Caller must free with ffi_free_string.
char *ffi_greet(const char *name);

// Free a string returned by FFI functions
void ffi_free_string(char *str);

// =============================================================================
// Opaque handle pattern (demonstrates resource management)
// =============================================================================

// Opaque handle to a Counter object
typedef struct Counter Counter;

// Create a new counter with initial value
Counter *ffi_counter_new(int32_t initial);

// Destroy a counter
void ffi_counter_free(Counter *counter);

// Get current value
int32_t ffi_counter_get(const Counter *counter);

// Increment and return new value
int32_t ffi_counter_increment(Counter *counter);

// Add n and return new value
int32_t ffi_counter_add(Counter *counter, int32_t n);

#ifdef __cplusplus
}
#endif

#endif // ALEPH_FFI_H
