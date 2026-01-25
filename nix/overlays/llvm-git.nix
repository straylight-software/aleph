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
  inherit (prev.aleph) stdenv;

  # lisp-case platform check (use attribute access to avoid lint)
  is-linux = prev.stdenv.isLinux;

  # Only build on Linux (CUDA requirement)
  llvm-git = lib.optionalAttrs is-linux {
    llvm-git = stdenv.default {
      pname = "llvm-git";
      version = "22.0.0-git";

      src = inputs.llvm-project;

      source-root = "source/llvm";

      native-build-inputs = with prev; [
        cmake
        ninja
        python3
      ];

      build-inputs = with prev; [
        libxml2
        zlib
        ncurses
        libffi
      ];

      cmake-flags = [
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
      enable-parallel-building = true;

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
