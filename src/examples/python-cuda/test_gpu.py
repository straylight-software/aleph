#!/usr/bin/env python3
"""Test the GPU module with CUDA kernels."""

import gpu_module
import sys


def test_scale_array():
    """Test GPU array scaling."""
    input_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = gpu_module.scale_array(input_arr, 2.0)
    expected = [2.0, 4.0, 6.0, 8.0, 10.0]

    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 1e-6, f"scale_array[{i}]: got {r}, expected {e}"
    print("  scale_array: pass")


def test_saxpy():
    """Test SAXPY: y = a*x + y."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    a = 2.0

    result = gpu_module.saxpy(y, a, x)
    # Expected: 2*[1,2,3,4,5] + [10,20,30,40,50] = [12,24,36,48,60]
    expected = [12.0, 24.0, 36.0, 48.0, 60.0]

    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 1e-6, f"saxpy[{i}]: got {r}, expected {e}"
    print("  saxpy: pass")


def test_dot_product():
    """Test GPU dot product."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = gpu_module.dot(a, b)
    # Expected: 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
    expected = 35.0

    assert abs(result - expected) < 1e-6, f"dot: got {result}, expected {expected}"
    print("  dot: pass")


def main():
    """Run all tests."""
    print("Python calling CUDA kernels via pybind11:")

    # Check CUDA availability
    if gpu_module.cuda_available():
        print(f"  CUDA device: {gpu_module.cuda_device_name()}")
        test_scale_array()
        test_saxpy()
        test_dot_product()
        print("all tests passed")
    else:
        print("  No CUDA devices available")
        print("  Skipping GPU tests (compilation succeeded)")
        # Exit successfully - we're testing the build, not the hardware
        sys.exit(0)


if __name__ == "__main__":
    main()
