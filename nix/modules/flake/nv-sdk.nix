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

      # Optionally: user-space NVIDIA driver bundle
      driver-pkg =
        if cfg.nvidia-driver != null then
          cfg.nvidia-driver
        else if pkgs ? linuxPackages && pkgs.linuxPackages ? nvidia_x11 then
          pkgs.linuxPackages.nvidia_x11
        else
          pkgs.nvidia_x11 or null;

      driver-pkg-final = if cfg.with-driver then driver-pkg else null;

      # passthru attributes (lisp-case local bindings for external API)
      cuda-version = nv-packages.cudatoolkit.version;
      gcc-version = pkgs.stdenv.cc.cc.version or "unknown";

      # symlinkJoin postBuild script
      post-build = ''
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRITICAL: lib64 -> lib symlink
        # Many NVIDIA tools expect lib64, but Nix uses lib
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if [ ! -e $out/lib64 ]; then
          ln -s lib $out/lib64
        fi

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CUDA 13 removed texture_fetch_functions.h (deprecated)
        # but clang's wrapper still expects it - symlink to replacement
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if [ ! -e $out/include/texture_fetch_functions.h ] && [ -e $out/include/texture_indirect_functions.h ]; then
          ln -s texture_indirect_functions.h $out/include/texture_fetch_functions.h
        fi

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # NVIDIA USER-SPACE DRIVER + NVML (and friends)
        # - Keep real driver libs available for runtime/debugging.
        # - Keep link-time stubs under $out/stubs like it was designed.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        DRIVER_PKG="${if driver-pkg-final != null then driver-pkg-final else ""}"

        if [ -n "$DRIVER_PKG" ]; then

          # Expose the full driver bundle under a stable prefix.
          if [ ! -e $out/driver ]; then
            ln -s "$DRIVER_PKG" $out/driver
          fi

          # Prefer putting driver runtime libs on the standard lib path.
          for libdir in "$DRIVER_PKG/lib" "$DRIVER_PKG/lib64"; do
            if [ -d "$libdir" ]; then
              for soname in \
                libcuda.so.1 libcuda.so \
                libnvidia-ml.so.1 libnvidia-ml.so \
                libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so \
                libnvidia-fatbinaryloader.so.1 libnvidia-fatbinaryloader.so \
                libnvidia-compiler.so.1 libnvidia-compiler.so \
                ; do
                if [ -e "$libdir/$soname" ] && [ ! -e "$out/lib64/$soname" ]; then
                  ln -s "$libdir/$soname" "$out/lib64/$soname"
                fi
              done
            fi
          done
        fi

        # Link-time stubs (compile/link against these; runtime comes from driver).
        mkdir -p $out/stubs/lib
        if [ ! -e $out/stubs/lib64 ]; then
          ln -s lib $out/stubs/lib64
        fi

        # Collect stubs from CUDA toolkit and (optionally) the driver package.
        STUB_DIRS=(
          "${nv-packages.cudatoolkit}/lib/stubs"
          "${nv-packages.cudatoolkit}/lib64/stubs"
        )

        if [ -n "$DRIVER_PKG" ]; then
          STUB_DIRS+=("$DRIVER_PKG/lib/stubs" "$DRIVER_PKG/lib64/stubs")
        fi

        for stubdir in "''${STUB_DIRS[@]}"; do
          if [ -d "$stubdir" ]; then
            # Symlink all stubs into $out/stubs/lib (flat).
            for f in "$stubdir"/*; do
              if [ -e "$f" ]; then
                ln -sf "$f" "$out/stubs/lib/$(basename "$f")"
              fi
            done
          fi
        done

        # ensure common link names exist in stubs dir...
        if [ -e "$out/stubs/lib/libcuda.so.1" ] && [ ! -e "$out/stubs/lib/libcuda.so" ]; then
          ln -s libcuda.so.1 $out/stubs/lib/libcuda.so
        fi

        if [ -e "$out/stubs/lib/libnvidia-ml.so.1" ] && [ ! -e "$out/stubs/lib/libnvidia-ml.so" ]; then
          ln -s libnvidia-ml.so.1 $out/stubs/lib/libnvidia-ml.so
        fi

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Create pkg-config files if they don't exist
        # (files are generated via writeText, no heredocs)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        mkdir -p $out/lib/pkgconfig
        sed "s|@PREFIX@|$out|g" ${cuda-pc} > $out/lib/pkgconfig/cuda.pc
        sed "s|@PREFIX@|$out|g" ${cudnn-pc} > $out/lib/pkgconfig/cudnn.pc
        sed "s|@PREFIX@|$out|g" ${tensorrt-pc} > $out/lib/pkgconfig/tensorrt.pc
        sed "s|@PREFIX@|$out|g" ${nccl-pc} > $out/lib/pkgconfig/nccl.pc

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # // version // manifest
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        cp ${sdk-manifest} $out/NVIDIA_SDK_MANIFEST
        echo "" >> $out/NVIDIA_SDK_MANIFEST
        echo "Contents:" >> $out/NVIDIA_SDK_MANIFEST
        echo "$(find $out -name '*.so' -o -name '*.a' 2>/dev/null | wc -l) libraries" >> $out/NVIDIA_SDK_MANIFEST
        echo "$(find $out/include -name '*.h' -o -name '*.hpp' -o -name '*.cuh' 2>/dev/null | wc -l) headers" >> $out/NVIDIA_SDK_MANIFEST

        echo "NVIDIA SDK build complete. See $out/NVIDIA_SDK_MANIFEST for details."
      '';

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
        "installPhase" = ''
          runHook preInstall

          mkdir -p $out/include
          cp -r include/cutlass $out/include/
          cp -r include/cute $out/include/

          # Tools and examples for reference
          mkdir -p $out/share/cutlass
          cp -r tools $out/share/cutlass/
          cp -r examples $out/share/cutlass/
          cp -r python $out/share/cutlass/

          echo "${cutlass-version}" > $out/CUTLASS_VERSION

          runHook postInstall
        '';

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

      cudnn-pc = write-text "cudnn.pc" ''
        prefix=@PREFIX@
        exec_prefix=''${prefix}
        libdir=''${prefix}/lib
        includedir=''${prefix}/include

        Name: cuDNN
        Description: NVIDIA CUDA Deep Neural Network library
        Version: ${nv-packages.cudnn.version}
        Libs: -L''${libdir} -lcudnn
        Cflags: -I''${includedir}
        Requires: cuda
      '';

      tensorrt-pc = write-text "tensorrt.pc" ''
        prefix=@PREFIX@
        exec_prefix=''${prefix}
        libdir=''${prefix}/lib
        includedir=''${prefix}/include

        Name: TensorRT
        Description: NVIDIA TensorRT inference library
        Version: ${nv-packages.tensorrt.version}
        Libs: -L''${libdir} -lnvinfer -lnvinfer_plugin -lnvonnxparser
        Cflags: -I''${includedir}
        Requires: cuda cudnn
      '';

      nccl-pc = write-text "nccl.pc" ''
        prefix=@PREFIX@
        exec_prefix=''${prefix}
        libdir=''${prefix}/lib
        includedir=''${prefix}/include

        Name: NCCL
        Description: NVIDIA Collective Communications Library
        Version: ${nv-packages.nccl.version}
        Libs: -L''${libdir} -lnccl
        Cflags: -I''${includedir}
        Requires: cuda
      '';

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
