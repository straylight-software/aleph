# overlays/llvm-git.nix — LLVM from git with SM120 Blackwell support
#
# Why build from source?
#   - nixpkgs clang's __clang_cuda_runtime_wrapper.h redefines uint3/dim3
#     as macros to __cuda_builtin_*_t types, breaking CCCL headers
#   - Building from source gives us a clean clang without the wrapper hacks
#   - SM120 (Blackwell) support requires bleeding edge LLVM
#
inputs: _final: prev:
let
  inherit (prev) lib;

  # Only build on Linux (CUDA requirement)
  llvm-git = lib.optionalAttrs prev.stdenv.isLinux {
    llvm-git = prev.stdenv.mkDerivation {
      pname = "llvm-git";
      version = "22.0.0-git";

      src = inputs.llvm-project;

      sourceRoot = "source/llvm";

      nativeBuildInputs = with prev; [
        cmake
        ninja
        python3
      ];

      buildInputs = with prev; [
        libxml2
        zlib
        ncurses
        libffi
      ];

      cmakeFlags = [
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AArch64"
        "-DLLVM_ENABLE_ASSERTIONS=OFF"
        "-DLLVM_INSTALL_UTILS=ON"
        "-DLLVM_BUILD_TOOLS=ON"
        "-DLLVM_INCLUDE_TESTS=OFF"
        "-DLLVM_INCLUDE_EXAMPLES=OFF"
        "-DLLVM_INCLUDE_DOCS=OFF"
        # Skip compiler-rt - CUDA doesn't require it and avoids i386 issues
      ];

      # LLVM is huge, enable parallel building
      enableParallelBuilding = true;

      meta = {
        description = "LLVM/Clang from git with CUDA 13 and SM120 Blackwell support";
        homepage = "https://llvm.org";
        license = lib.licenses.ncsa;
        platforms = lib.platforms.linux;
      };
    };
  };
in
llvm-git
