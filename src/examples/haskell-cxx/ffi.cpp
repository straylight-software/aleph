// ffi.cpp
// C++ implementation of FFI functions for Haskell
//
// This demonstrates calling C++ from Haskell via the FFI.
// All exported functions use extern "C" linkage for ABI compatibility.

#include "ffi.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <string>

// =============================================================================
// Simple arithmetic
// =============================================================================

extern "C" int32_t ffi_add(int32_t a, int32_t b) { return a + b; }

extern "C" int32_t ffi_multiply(int32_t a, int32_t b) { return a * b; }

// =============================================================================
// Vector operations
// =============================================================================

extern "C" double ffi_dot_product(const double *a, const double *b,
                                  size_t len) {
  double result = 0.0;
  for (size_t i = 0; i < len; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

extern "C" double ffi_norm(const double *v, size_t len) {
  return std::sqrt(ffi_dot_product(v, v, len));
}

extern "C" void ffi_scale(double *v, size_t len, double scalar) {
  for (size_t i = 0; i < len; ++i) {
    v[i] *= scalar;
  }
}

// =============================================================================
// String operations
// =============================================================================

extern "C" char *ffi_greet(const char *name) {
  std::string greeting = "Hello from C++, " + std::string(name) + "!";

  // Allocate with malloc so Haskell can free with ffi_free_string
  char *result = static_cast<char *>(std::malloc(greeting.size() + 1));
  if (result) {
    std::strcpy(result, greeting.c_str());
  }
  return result;
}

extern "C" void ffi_free_string(char *str) { std::free(str); }

// =============================================================================
// Counter (opaque handle pattern)
// =============================================================================

// C++ class behind the opaque handle
class CounterImpl {
public:
  explicit CounterImpl(int32_t initial) : value_(initial) {}

  int32_t get() const { return value_; }
  int32_t increment() { return ++value_; }
  int32_t add(int32_t n) {
    value_ += n;
    return value_;
  }

private:
  int32_t value_;
};

// The Counter struct is just an alias for the C++ class
struct Counter : public CounterImpl {
  using CounterImpl::CounterImpl;
};

extern "C" Counter *ffi_counter_new(int32_t initial) {
  return new Counter(initial);
}

extern "C" void ffi_counter_free(Counter *counter) { delete counter; }

extern "C" int32_t ffi_counter_get(const Counter *counter) {
  return counter->get();
}

extern "C" int32_t ffi_counter_increment(Counter *counter) {
  return counter->increment();
}

extern "C" int32_t ffi_counter_add(Counter *counter, int32_t n) {
  return counter->add(n);
}
