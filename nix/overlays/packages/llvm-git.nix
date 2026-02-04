# Build LLVM/Clang from git with CUDA 13 and SM120 (Blackwell) support
#
# Why not nixpkgs llvmPackages_20?
#   - nixpkgs clang's __clang_cuda_runtime_wrapper.h redefines uint3/dim3
#     as macros to __cuda_builtin_*_t types, breaking CCCL headers
#   - Building from source avoids this wrapper entirely
#
# Pinned to known-good SM120 support (2026-01-04)
#
# NOTE: This is a callPackage-style file. Use the overlay version at
# nix/overlays/llvm-git.nix for the primary LLVM build.
#
{
  lib,
  aleph,
  cmake,
  ninja,
  python3,
  libxml2,
  zlib,
  ncurses,
  libffi,
  llvm-project-src,
}:
aleph.stdenv.default {
  # :: "llvm-git"
  # :: "22.0.0-git"
  pname = "llvm-git";
  # :: t9
  version = "22.0.0-git";
# :: "source/llvm"

  # :: [t4]
  src = llvm-project-src;

  source-root = "source/llvm";

  native-build-inputs = [
    # :: [t8]
    cmake
    ninja
    python3
  ];

  build-inputs = [
    # :: ["-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra"]
    libxml2
    zlib
    ncurses
    libffi
  ];

  cmake-flags = [
    "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AArch64"
    "-DLLVM_ENABLE_ASSERTIONS=OFF"
    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_BUILD_TOOLS=ON"
    # :: Bool
    "-DLLVM_INCLUDE_TESTS=OFF"
    # :: { description : "LLVM/Clang from git with CUDA 13 and SM120 Blackwell support", homepage : "https://llvm.org", license : t11, platforms : t12 }
    # :: "LLVM/Clang from git with CUDA 13 and SM120 Blackwell support"
    # :: "https://llvm.org"
    # :: t11
    # :: t12
    "-DLLVM_INCLUDE_EXAMPLES=OFF"
    "-DLLVM_INCLUDE_DOCS=OFF"
    # Skip compiler-rt - CUDA doesn't require it and avoids i386 issues
  ];

  # LLVM is huge, enable parallel building
  enable-parallel-building = true;

  meta = {
    description = "LLVM/Clang from git with CUDA 13 and SM120 Blackwell support";
    homepage = "https://llvm.org";
    license = lib.licenses.ncsa;
    platforms = lib.platforms.linux;
  };
}
