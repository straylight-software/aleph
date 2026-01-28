#!/usr/bin/env bash
# Test NVIDIA SDK structure

echo "Testing NVIDIA SDK structure..."

# Check for critical CUDA headers
echo "Checking for cuda.h..."
if [ ! -f "@nvidiaSdk@/include/cuda.h" ]; then
  echo "x FAILED: cuda.h not found"
  exit 1
fi
echo "v cuda.h found"

# Check for CUTLASS cute tensor headers
echo "Checking for cute/tensor.hpp..."
if [ ! -f "@nvidiaSdk@/include/cute/tensor.hpp" ]; then
  echo "x FAILED: cute/tensor.hpp not found"
  exit 1
fi
echo "v cute/tensor.hpp found"

# Check for lib64 symlink or directory
echo "Checking for lib64..."
if [ ! -e "@nvidiaSdk@/lib64" ]; then
  echo "x FAILED: lib64 not found"
  exit 1
fi
echo "v lib64 exists"

# Check for CCCL structure (CUDA C++ Core Libraries)
echo "Checking for CCCL structure..."
if [ ! -d "@nvidiaSdk@/include/cuda" ] &&
  [ ! -d "@nvidiaSdk@/include/thrust" ] &&
  [ ! -d "@nvidiaSdk@/include/cub" ]; then
  echo "x FAILED: CCCL structure (cuda/thrust/cub) not found"
  exit 1
fi
echo "v CCCL structure exists"

# Check for cudart library
echo "Checking for CUDA runtime library..."
if [ ! -f "@nvidiaSdk@/lib/libcudart.so" ] &&
  [ ! -f "@nvidiaSdk@/lib64/libcudart.so" ]; then
  echo "x FAILED: libcudart.so not found"
  exit 1
fi
echo "v libcudart.so found"

mkdir -p "$out"
echo "SUCCESS" >"$out/SUCCESS"
echo "All NVIDIA SDK structure checks passed" >>"$out/SUCCESS"
echo "  - cuda.h exists" >>"$out/SUCCESS"
echo "  - cute/tensor.hpp exists" >>"$out/SUCCESS"
echo "  - lib64 exists" >>"$out/SUCCESS"
echo "  - CCCL structure exists" >>"$out/SUCCESS"
echo "  - libcudart.so exists" >>"$out/SUCCESS"
