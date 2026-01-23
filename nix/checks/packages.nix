# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH-NAUGHT PACKAGE TESTS                          ║
# ║                                                                            ║
# ║  Tests for packages exposed by aleph-naught.                               ║
# ║  Ensures packages are properly built and usable.                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # ══════════════════════════════════════════════════════════════════════════
  # TEST: mdspan-installation
  # ══════════════════════════════════════════════════════════════════════════
  # Verify that mdspan headers are properly installed and can be used
  # to compile a C++23 program using std::mdspan

  test-mdspan-installation = pkgs.stdenv.mkDerivation {
    name = "test-mdspan-installation";

    src = pkgs.writeTextDir "test.cpp" ''
      #include <mdspan>
      #include <vector>
      #include <cassert>

      int main() {
        // Test std::mdspan is available (not std::experimental::mdspan)
        std::vector<int> data = {1, 2, 3, 4, 5, 6};
        
        // Create a 2D mdspan view of the data (2x3 matrix)
        std::mdspan<int, std::dextents<size_t, 2>> mat(
          data.data(), 
          2, 3
        );
        
        // Verify dimensions
        assert(mat.extent(0) == 2);
        assert(mat.extent(1) == 3);
        
        // Verify data access (parentheses needed to avoid comma-in-macro issue)
        assert((mat[0, 0] == 1));
        assert((mat[0, 1] == 2));
        assert((mat[0, 2] == 3));
        assert((mat[1, 0] == 4));
        assert((mat[1, 1] == 5));
        assert((mat[1, 2] == 6));
        
        return 0;
      }
    '';

    nativeBuildInputs = [
      pkgs.gcc15
      pkgs.mdspan
    ];

    buildPhase = ''
      echo "Building mdspan test program..."
      g++ -std=c++23 -I${pkgs.mdspan}/include test.cpp -o test
    '';

    doCheck = true;
    checkPhase = ''
      echo "Running mdspan test..."
      ./test
      echo "✓ mdspan test passed"
    '';

    installPhase = ''
      mkdir -p $out
      echo "SUCCESS" > $out/SUCCESS
      echo "mdspan C++23 headers work correctly" >> $out/SUCCESS
    '';

    meta = {
      description = "Test that mdspan C++23 headers are properly installed and usable";
    };
  };

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: nvidia-sdk-structure (Linux-only)
  # ══════════════════════════════════════════════════════════════════════════
  # Verify that the NVIDIA SDK has the expected structure and critical headers

  test-nvidia-sdk-structure =
    pkgs.runCommand "test-nvidia-sdk-structure"
      {
        nativeBuildInputs = [ pkgs.nvidia-sdk ];
      }
      ''
        echo "Testing NVIDIA SDK structure..."

        # Check for critical CUDA headers
        echo "Checking for cuda.h..."
        if [ ! -f "${pkgs.nvidia-sdk}/include/cuda.h" ]; then
          echo "✗ FAILED: cuda.h not found"
          exit 1
        fi
        echo "✓ cuda.h found"

        # Check for CUTLASS cute tensor headers
        echo "Checking for cute/tensor.hpp..."
        if [ ! -f "${pkgs.nvidia-sdk}/include/cute/tensor.hpp" ]; then
          echo "✗ FAILED: cute/tensor.hpp not found"
          exit 1
        fi
        echo "✓ cute/tensor.hpp found"

        # Check for lib64 symlink or directory
        echo "Checking for lib64..."
        if [ ! -e "${pkgs.nvidia-sdk}/lib64" ]; then
          echo "✗ FAILED: lib64 not found"
          exit 1
        fi
        echo "✓ lib64 exists"

        # Check for CCCL structure (CUDA C++ Core Libraries)
        echo "Checking for CCCL structure..."
        if [ ! -d "${pkgs.nvidia-sdk}/include/cuda" ] && \
           [ ! -d "${pkgs.nvidia-sdk}/include/thrust" ] && \
           [ ! -d "${pkgs.nvidia-sdk}/include/cub" ]; then
          echo "✗ FAILED: CCCL structure (cuda/thrust/cub) not found"
          exit 1
        fi
        echo "✓ CCCL structure exists"

        # Check for cudart library
        echo "Checking for CUDA runtime library..."
        if [ ! -f "${pkgs.nvidia-sdk}/lib/libcudart.so" ] && \
           [ ! -f "${pkgs.nvidia-sdk}/lib64/libcudart.so" ]; then
          echo "✗ FAILED: libcudart.so not found"
          exit 1
        fi
        echo "✓ libcudart.so found"

        mkdir -p $out
        echo "SUCCESS" > $out/SUCCESS
        echo "All NVIDIA SDK structure checks passed" >> $out/SUCCESS
        echo "  - cuda.h exists" >> $out/SUCCESS
        echo "  - cute/tensor.hpp exists" >> $out/SUCCESS
        echo "  - lib64 exists" >> $out/SUCCESS
        echo "  - CCCL structure exists" >> $out/SUCCESS
        echo "  - libcudart.so exists" >> $out/SUCCESS
      '';

in
{
  # Always include mdspan test
  inherit test-mdspan-installation;

  # Only include NVIDIA SDK test on Linux
}
// lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit test-nvidia-sdk-structure;
}
