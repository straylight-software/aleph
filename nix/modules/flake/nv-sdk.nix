# nix/modules/flake/nv-sdk.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // nvidia sdk //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The matrix has its roots in primitive arcade games, in early graphics
#      programs and military experimentation with cranial jacks."
#
#                                                         — Neuromancer
#
# Comprehensive NVIDIA SDK based on nixpkgs nv-packages.
# Alternative to aleph-nvidia-sdk for standard nixpkgs support.
#
# We say "nv" not "cuda". See: docs/languages/nix/philosophy/nvidia-not-cuda.md
#
# Usage in flake.nix:
#   imports = [ ./nix/flake-modules/nixpkgs-nvidia.nix ];
#
#   perSystem = { ... }: {
#     nv.sdk = {
#       sdk-version = "13";
#       with-driver = true;
#     };
#   };
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, flake-parts-lib, ... }:
let
  # ──────────────────────────────────────────────────────────────────────────────
  # // lisp-case aliases for lib.* functions //
  # ──────────────────────────────────────────────────────────────────────────────
  mk-enable-option = lib.mkEnableOption;
  mk-option = lib.mkOption;
  mk-if = lib.mkIf;
  mk-per-system-option = flake-parts-lib.mkPerSystemOption;
  null-or = lib.types.nullOr;
in
{
  _class = "flake";

  # ────────────────────────────────────────────────────────────────────────────
  # // options //
  # ────────────────────────────────────────────────────────────────────────────

  options.perSystem = mk-per-system-option {
    options.nv.sdk = {
      enable = mk-enable-option "nixpkgs-based NVIDIA SDK" // {
        default = false;
      };

      sdk-version = mk-option {
        type = lib.types.enum [
          "12_9"
          "13"
        ];
        default = "13";
        description = "SDK version to use from nixpkgs nv-packages";
      };

      nvidia-driver = mk-option {
        type = null-or lib.types.package;
        default = null;
        description = ''
          User-space NVIDIA driver package (libcuda, libnvidia-ml, stubs).
          If null, will try to auto-detect from pkgs.linuxPackages.nvidia_x11.
        '';
      };

      with-driver = mk-option {
        type = lib.types.bool;
        default = true;
        description = "Include NVIDIA driver in the SDK bundle";
      };

      # NOTE: Custom gcc option removed - rebuilding cudaPackages with different
      # gcc is complex and breaks nixpkgs bootstrap assertions. The SDK exposes
      # passthru.gcc so consumers can detect the compiler it was built with.

      sdk = mk-option {
        type = lib.types.package;
        "readOnly" = true;
        description = "The computed NVIDIA SDK package";
      };

      cutlass = mk-option {
        type = lib.types.package;
        "readOnly" = true;
        description = "CUTLASS package (CUDA Templates for Linear Algebra Subroutines)";
      };
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // config //
  # ────────────────────────────────────────────────────────────────────────────

  config.perSystem =
    {
      config,
      pkgs,
      lib,
      ...
    }:
    let
      # ────────────────────────────────────────────────────────────────────────────
      # // lisp-case aliases for pkgs.* and lib.* functions //
      # ────────────────────────────────────────────────────────────────────────────
      mk-derivation = pkgs.stdenv.mkDerivation;
      fetch-from-github = pkgs.fetchFromGitHub;
      symlink-join = pkgs.symlinkJoin;
      write-text = pkgs.writeText;

      # cutlass version
      cutlass-version = "4.3.3";

      cfg = config.nv.sdk;

      # Note: We use pkgs directly without overriding stdenv/gcc.
      # Rebuilding entire nv-packages with a different gcc is complex and breaks
      # nixpkgs bootstrap assertions. Instead, we expose what gcc the SDK was built
      # with via passthru, so consumers can detect mismatches and make informed decisions.
      nv-packages =
        if cfg.sdk-version == "13" then
          pkgs.cudaPackages_13
        else if cfg.sdk-version == "12" then
          pkgs.cudaPackages_12
        else if cfg.sdk-version == "12_4" then
          pkgs.cudaPackages_12_4
        else if cfg.sdk-version == "12_8" then
          pkgs.cudaPackages_12_8
        else
          pkgs.cudaPackages;

      # User-space NVIDIA driver bundle (required when with-driver = true)
      driver-pkg =
        if cfg.nvidia-driver != null then
          cfg.nvidia-driver
        else if pkgs ? linuxPackages && pkgs.linuxPackages ? nvidia_x11 then
          pkgs.linuxPackages.nvidia_x11
        else
          pkgs.nvidia_x11;

      driver-pkg-final = if cfg.with-driver then driver-pkg else null;

      # passthru attributes (lisp-case local bindings for external API)
      cuda-version = nv-packages.cudatoolkit.version;
      gcc-version = pkgs.stdenv.cc.cc.version;

      # symlinkJoin postBuild script (loaded from external file)
      post-build-script = builtins.readFile ./nv-sdk/scripts/sdk-post-build.sh;
      post-build =
        builtins.replaceStrings
          [
            "@driverPkg@"
            "@cudatoolkit@"
            "@cudaPc@"
            "@cudnnPc@"
            "@tensorrtPc@"
            "@ncclPc@"
            "@sdkManifest@"
          ]
          [
            (if driver-pkg-final != null then "${driver-pkg-final}" else "")
            "${nv-packages.cudatoolkit}"
            "${cuda-pc}"
            "${cudnn-pc}"
            "${tensorrt-pc}"
            "${nccl-pc}"
            "${sdk-manifest}"
          ]
          post-build-script;

      # ──────────────────────────────────────────────────────────────────────────
      # // cutlass //
      # ──────────────────────────────────────────────────────────────────────────

      # CUTLASS 4.3.3 - fetch directly since nixpkgs may not have latest
      cutlass-latest = mk-derivation {
        pname = "cutlass";
        version = cutlass-version;

        src = fetch-from-github {
          owner = "NVIDIA";
          repo = "cutlass";
          tag = "v${cutlass-version}";
          hash = "sha256-uOfSEjbwn/edHEgBikC9wAarn6c6T71ebPg74rv2qlw=";
        };

        "dontBuild" = true;
        "dontConfigure" = true;
        "installPhase" = builtins.replaceStrings [ "@version@" ] [ cutlass-version ] (
          builtins.readFile ./nv-sdk/scripts/cutlass-install.sh
        );

        meta = {
          description = "CUDA Templates for Linear Algebra Subroutines";
          homepage = "https://github.com/NVIDIA/cutlass";
          license = lib.licenses.bsd3;
          platforms = lib.platforms.unix;
        };
      };

      cutlass-package = nv-packages.cutlass or cutlass-latest;

      # ──────────────────────────────────────────────────────────────────────────
      # // pkg-config files //
      # ──────────────────────────────────────────────────────────────────────────

      cuda-pc = write-text "cuda.pc" ''
        prefix=@PREFIX@
        exec_prefix=''${prefix}
        libdir=''${prefix}/lib
        includedir=''${prefix}/include

        Name: CUDA
        Description: NVIDIA CUDA Toolkit
        Version: ${nv-packages.cudatoolkit.version}
        Libs: -L''${libdir} -lcudart -lcuda
        Cflags: -I''${includedir}
      '';

      cudnn-pc = write-text "cudnn.pc" (
        builtins.replaceStrings [ "@VERSION@" ] [ nv-packages.cudnn.version ] (
          builtins.readFile ./nv-sdk/scripts/cudnn.pc.in
        )
      );

      tensorrt-pc = write-text "tensorrt.pc" (
        builtins.replaceStrings [ "@VERSION@" ] [ nv-packages.tensorrt.version ] (
          builtins.readFile ./nv-sdk/scripts/tensorrt.pc.in
        )
      );

      nccl-pc = write-text "nccl.pc" (
        builtins.replaceStrings [ "@VERSION@" ] [ nv-packages.nccl.version ] (
          builtins.readFile ./nv-sdk/scripts/nccl.pc.in
        )
      );

      sdk-manifest = write-text "NVIDIA_SDK_MANIFEST" ''
        NVIDIA SDK for Nix (nixpkgs-based)

        CUDA Toolkit: ${nv-packages.cudatoolkit.version}
        cuDNN: ${nv-packages.cudnn.version}
        TensorRT: ${nv-packages.tensorrt.version}
        NCCL: ${nv-packages.nccl.version}
        Driver bundle: ${if driver-pkg-final != null then "included" else "not included"}
        CUTLASS: ${cutlass-package.version or "included"}
      '';

      # ──────────────────────────────────────────────────────────────────────────
      # // sdk bundle //
      # ──────────────────────────────────────────────────────────────────────────

      nvidia-sdk = symlink-join {
        name = "nvidia-sdk-cuda${cuda-version}";

        paths = [

          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          #                                // CUDATOOLKIT
          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

          nv-packages.cudatoolkit # `nvcc`, `cuda` runtime, etc.
          nv-packages.cuda_cudart # CUDA runtime library
          nv-packages.cuda_crt # CUDA CRT headers
          nv-packages.cuda_nvcc # NVIDIA CUDA Compiler
          nv-packages.cuda_nvrtc # Runtime Compilation
          nv-packages.cuda_nvml_dev # NVIDIA Management Library

          # nv-packages.cuda_nvprof         # Profiler (unsupported on x86_64-linux in CUDA 13)

          nv-packages.cuda_cupti # Profiling Tools Interface
          nv-packages.cuda_cuobjdump # Object dump utility
          nv-packages.cuda_cuxxfilt # C++ name demangler
          nv-packages.cuda_gdb # CUDA debugger
          nv-packages.cuda_sanitizer_api # Compute Sanitizer
          nv-packages.cuda_nvdisasm # Disassembler

          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          #                                // MATHEMATICS
          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

          nv-packages.libcublas # BLAS
          nv-packages.libcufft # FFT
          nv-packages.libcurand # Random number generation
          nv-packages.libcusolver # Dense/sparse direct solvers
          nv-packages.libcusparse # Sparse matrix operations
          nv-packages.libnvjitlink # JIT linker

          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          #                              // DEEP LEARNING
          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

          nv-packages.cudnn # Deep Neural Network library
          nv-packages.nccl # Multi-GPU collective communications

          nv-packages.tensorrt # myelin: jenson's razor.

          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          #                                   // OPTIONAL
          # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ]
        ++ lib.optional (nv-packages ? libcufile) nv-packages.libcufile # GPUDirect Storage
        ++ lib.optional (nv-packages ? libnpp) nv-packages.libnpp # Image/signal processing
        ++ lib.optional (nv-packages ? libcusparselt) nv-packages.libcusparselt # Sparse LT
        ++ lib.optional (nv-packages ? cuda_nsight) nv-packages.cuda_nsight # Nsight IDE
        ++ lib.optional (nv-packages ? nsight_compute) nv-packages.nsight_compute
        ++ lib.optional (nv-packages ? nsight_systems) nv-packages.nsight_systems

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #                                    // CUTLASS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ++ [ cutlass-package ]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #                        // SYSTEM DEPENDENCIES
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ++ [
          pkgs.libfabric # High-performance fabric
          pkgs.numactl # NUMA support
          pkgs.rdma-core # GPUDirect RDMA
          pkgs.stdenv.cc.cc.lib # libstdc++
          pkgs.ucx # Unified Communication X
          pkgs.zlib # compression
          pkgs.zstd # compression (newer)
        ]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #                                        // MPI
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ++ lib.optional (pkgs ? openmpi) pkgs.openmpi;

        "postBuild" = post-build;

        "passthru" = {

          # expose what this SDK was built with for cache invalidation and compatibility checking
          inherit (pkgs) stdenv;
          inherit (pkgs.stdenv) cc;
          inherit nv-packages;
          inherit cutlass-package;

          "cudaVersion" = cuda-version;
          "gccVersion" = gcc-version;

          gcc = pkgs.stdenv.cc.cc;
        };

        meta = {
          description = "Comprehensive NVIDIA CUDA/ML SDK for Nix (nixpkgs-based)";
          homepage = "https://developer.nvidia.com/cuda-toolkit";
          license = lib.licenses.unfree;
          platforms = [
            "x86_64-linux"
            "aarch64-linux"
          ];
        };
      };
    in
    mk-if cfg.enable {
      cuda.nixpkgs = {
        sdk = nvidia-sdk;
        cutlass = cutlass-package;
      };

      packages = {
        nvidia-sdk-cuda = nvidia-sdk;
        cutlass = cutlass-package;
      };
    };
}
