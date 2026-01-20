# nvidia-sdk packages
#
# NVIDIA libraries extracted via typed Haskell scripts.
# Wheels from PyPI (preferred, no redistribution), containers for CUDA toolkit.
#
# Architecture:
#   - Wheels are fetched as FODs (fetchurl with hash), then extracted via Haskell
#   - Container rootfs comes from default.nix FOD, extracted via Haskell
#
# Usage:
#   pkgs.nvidia-nccl           # NCCL from wheel
#   pkgs.nvidia-cudnn          # cuDNN from wheel
#   pkgs.nvidia-tensorrt       # TensorRT from wheel
#   pkgs.nvidia-cutensor       # cuTensor from wheel
#   pkgs.nvidia-cusparselt     # cuSPARSELt from wheel
#   pkgs.nvidia-cutlass        # CUTLASS headers
#   pkgs.nvidia-cuda-toolkit   # Full CUDA toolkit from container
#   pkgs.nvidia-tritonserver   # Triton Inference Server from container
#
final: prev:
let
  inherit (prev) lib fetchurl;
  inherit (prev.stdenv.hostPlatform) system;

  # ════════════════════════════════════════════════════════════════════════════
  # Runtime dependencies for autoPatchelfHook
  # ════════════════════════════════════════════════════════════════════════════

  runtimeDeps = with final; [
    stdenv.cc.cc.lib # libstdc++
    zlib
  ];

  extendedRuntimeDeps =
    runtimeDeps
    ++ (with final; [
      openssl
      curl
      expat
      libxml2
      ncurses
      gmp
      rdma-core
    ]);

  # libxml2 2.9.14 - needed because the container uses an older version
  libxml2-legacy = final.libxml2.overrideAttrs (_old: rec {
    version = "2.9.14";
    src = fetchurl {
      url = "https://download.gnome.org/sources/libxml2/${lib.versions.majorMinor version}/libxml2-${version}.tar.xz";
      sha256 = "sha256-YNdKJX0czsBHXnScui8hVZ5IE577pv8oIkNXx8eY3+4=";
    };
  });

  # Comprehensive runtime deps for tritonserver (mirrors libmodern-nvidia-sdk)
  tritonRuntimeDeps = with final; [
    # Core
    stdenv.cc.cc.lib
    zlib
    python312
    # Network/SSL
    openssl
    curl
    # Serialization
    expat
    libxml2-legacy
    # Terminal
    ncurses
    readline
    # Math/Compression
    gmp
    bzip2
    xz
    lz4
    # RDMA
    rdma-core
    # Triton framework deps
    abseil-cpp
    boost
    grpc
    protobuf
    re2
    libevent
    numactl
    openmpi
    rapidjson
    gperftools
    # Archive
    libarchive
    # Unicode
    icu
    # FFI
    libffi
    # GLib ecosystem
    glib
    pcre2
    # System libs (from working libmodern-nvidia-sdk)
    systemd
    libgcrypt
    libgpg-error
    libcap
    libcap_ng
    audit
    libselinux
    libsemanage
    libsepol
    pcre
    libkrb5
    keyutils
    dbus
    pam
    e2fsprogs.dev
    util-linux
    libmd
    libbsd
    gdbm
    db
    tzdata
    libxcrypt
    # Additional deps
    nettle
    acl
    cyrus_sasl
    gnutls
    libssh
    openldap
    rtmpdump
    libuuid
  ];

  # Common ignored dependencies (driver libs provided at runtime)
  commonIgnoredDeps = [
    "libcuda.so.1"
    "libnvidia-ml.so.1"
    # CUDA runtime libs (cross-references between packages)
    "libcudart.so.13"
    "libcublas.so.13"
    "libcublasLt.so.13"
    "libcufft.so.13"
    "libcurand.so.13"
    "libcusolver.so.13"
    "libcusparse.so.13"
    "libnvrtc.so.13"
    "libnccl.so.2"
  ];

  # ════════════════════════════════════════════════════════════════════════════
  # Wheel definitions (from Aleph.Script.Nvidia.Wheel, mirrored here for FOD)
  # ════════════════════════════════════════════════════════════════════════════

  wheels = {
    nccl = {
      version = "2.28.9";
      url = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
      hash = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI=";
      libPath = "nvidia/nccl/lib";
      includePath = "nvidia/nccl/include";
    };
    cudnn = {
      version = "9.17.0.29";
      url = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl";
      hash = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo=";
      libPath = "nvidia/cudnn/lib";
      includePath = "nvidia/cudnn/include";
    };
    tensorrt = {
      version = "10.14.1.48";
      url = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY=";
      libPath = "tensorrt_libs";
      includePath = null;
    };
    cutensor = {
      version = "2.4.1";
      url = "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl";
      hash = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg=";
      libPath = "cutensor/lib";
      includePath = "cutensor/include";
    };
    cusparselt = {
      version = "0.8.1";
      url = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl";
      hash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA=";
      libPath = "nvidia/cusparselt/lib";
      includePath = "nvidia/cusparselt/include";
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # mkWheelPkg - extract library from PyPI wheel
  # ════════════════════════════════════════════════════════════════════════════
  #
  # Two-phase:
  #   1. fetchurl downloads the wheel (FOD with hash)
  #   2. Derivation extracts and patches

  mkWheelPkg =
    {
      pname,
      wheelInfo,
      runtimeInputs ? runtimeDeps,
      ignoreMissingDeps ? [ ],
      meta ? { },
    }:
    prev.stdenv.mkDerivation {
      inherit pname;
      inherit (wheelInfo) version;

      src = fetchurl {
        inherit (wheelInfo) url hash;
      };

      nativeBuildInputs = [
        final.autoPatchelfHook
        final.unzip
        final.patchelf
        final.findutils
      ];

      buildInputs = runtimeInputs;

      autoPatchelfIgnoreMissingDeps = commonIgnoredDeps ++ ignoreMissingDeps;

      dontConfigure = true;
      dontBuild = true;

      unpackPhase = ''
        runHook preUnpack
        unzip $src -d unpacked
        runHook postUnpack
      '';

      installPhase =
        let
          libPath = wheelInfo.libPath or null;
          includePath = wheelInfo.includePath or null;
        in
        ''
          runHook preInstall
          mkdir -p $out

          ${lib.optionalString (libPath != null) ''
            if [ -d "unpacked/${libPath}" ]; then
              mkdir -p $out/lib
              cp -r unpacked/${libPath}/* $out/lib/
            fi
          ''}

          ${lib.optionalString (includePath != null) ''
            if [ -d "unpacked/${includePath}" ]; then
              mkdir -p $out/include
              cp -r unpacked/${includePath}/* $out/include/
            fi
          ''}

          # Create lib64 symlink for compatibility
          [ -d $out/lib ] && [ ! -e $out/lib64 ] && ln -s lib $out/lib64 || true

          # Make writable for patchelf
          chmod -R u+w $out 2>/dev/null || true

          # Patch RPATH for portability (before autoPatchelfHook runs)
          find $out -name "*.so*" -type f | while read f; do
            patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64' "$f" 2>/dev/null || true
          done

          runHook postInstall
        '';

      preFixup = ''
        addAutoPatchelfSearchPath $out/lib
      '';

      inherit meta;
    };

  # ════════════════════════════════════════════════════════════════════════════
  # mkContainerPkg - extract from container rootfs using Haskell script
  # ════════════════════════════════════════════════════════════════════════════

  tritonRootfs = prev.nvidia-sdk-ngc-rootfs or null;

  # Haskell extraction script
  nvidia-sdk-script = final.straylight.script.compiled.nvidia-sdk;

  mkContainerPkg =
    {
      pname,
      version,
      rootfs,
      extractMode, # "cuda" | "triton" | "runtime"
      runtimeInputs ? extendedRuntimeDeps,
      ignoreMissingDeps ? [ ],
      postExtract ? "",
      meta ? { },
    }:
    prev.stdenv.mkDerivation {
      inherit pname version;

      # No src - we use the rootfs directly
      dontUnpack = true;
      dontConfigure = true;
      dontBuild = true;

      nativeBuildInputs = [
        final.autoPatchelfHook
        final.makeWrapper
        final.patchelf
        final.file
        nvidia-sdk-script
      ];

      buildInputs = runtimeInputs;

      autoPatchelfIgnoreMissingDeps =
        commonIgnoredDeps
        ++ [
          "libpython3.8.so.1.0"
          "libpython3.9.so.1.0"
          "libpython3.10.so.1.0"
          "libpython3.11.so.1.0"
          "libpython3.12.so.1.0"
        ]
        ++ ignoreMissingDeps;

      installPhase = ''
        runHook preInstall
        nvidia-sdk ${extractMode} ${rootfs} $out
        ${postExtract}
        runHook postInstall
      '';

      preFixup = ''
        # Add search paths for autoPatchelf
        [ -d $out/lib ] && addAutoPatchelfSearchPath $out/lib
        [ -d $out/lib64 ] && addAutoPatchelfSearchPath $out/lib64
        [ -d $out/nvvm/lib64 ] && addAutoPatchelfSearchPath $out/nvvm/lib64
        [ -d $out/tensorrt_llm/lib ] && addAutoPatchelfSearchPath $out/tensorrt_llm/lib

        # Build library path from runtime inputs
        local libPaths="$out/lib"
        [ -d $out/lib64 ] && libPaths="$libPaths:$out/lib64"
        [ -d $out/nvvm/lib64 ] && libPaths="$libPaths:$out/nvvm/lib64"
        [ -d $out/tensorrt_llm/lib ] && libPaths="$libPaths:$out/tensorrt_llm/lib"
        libPaths="$libPaths:${lib.makeLibraryPath runtimeInputs}"

        echo "Setting RPATH on ELF files before autoPatchelf..."

        # Pre-patch all ELF files with correct RPATH before autoPatchelf runs
        # This helps autoPatchelf find dependencies and prevents silent failures
        find $out -type f 2>/dev/null | while read -r f; do
          if file "$f" 2>/dev/null | grep -q "ELF"; then
            # Set interpreter for executables
            if file "$f" | grep -q "ELF.*executable"; then
              patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$f" 2>/dev/null || true
            fi
            # Set RPATH for all ELF files
            patchelf --set-rpath "$libPaths" "$f" 2>/dev/null || true
          fi
        done
      '';

      postFixup = ''
        # Wrap executables with proper environment after autoPatchelf
        local libPaths="$out/lib"
        [ -d $out/lib64 ] && libPaths="$libPaths:$out/lib64"
        [ -d $out/tensorrt_llm/lib ] && libPaths="$libPaths:$out/tensorrt_llm/lib"
        libPaths="$libPaths:${lib.makeLibraryPath runtimeInputs}"

        for exe in $out/bin/*; do
          if [ -f "$exe" ] && [ -x "$exe" ]; then
            wrapProgram "$exe" \
              --prefix LD_LIBRARY_PATH : "$libPaths" \
              --prefix PYTHONPATH : "$out/python" 2>/dev/null || true
          fi
        done
      '';

      inherit meta;
    };

in

# ══════════════════════════════════════════════════════════════════════════════
# Wheel-based packages (preferred - no redistribution issues)
# ══════════════════════════════════════════════════════════════════════════════

lib.optionalAttrs (system == "x86_64-linux") {
  nvidia-nccl = mkWheelPkg {
    pname = "nvidia-nccl";
    wheelInfo = wheels.nccl;
    meta = {
      description = "NVIDIA NCCL ${wheels.nccl.version} (from PyPI)";
      homepage = "https://developer.nvidia.com/nccl";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cudnn = mkWheelPkg {
    pname = "nvidia-cudnn";
    wheelInfo = wheels.cudnn;
    meta = {
      description = "NVIDIA cuDNN ${wheels.cudnn.version} (from PyPI)";
      homepage = "https://developer.nvidia.com/cudnn";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-tensorrt = mkWheelPkg {
    pname = "nvidia-tensorrt";
    wheelInfo = wheels.tensorrt;
    meta = {
      description = "NVIDIA TensorRT ${wheels.tensorrt.version} (from PyPI)";
      homepage = "https://developer.nvidia.com/tensorrt";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cutensor = mkWheelPkg {
    pname = "nvidia-cutensor";
    wheelInfo = wheels.cutensor;
    meta = {
      description = "NVIDIA cuTensor ${wheels.cutensor.version} (from PyPI)";
      homepage = "https://developer.nvidia.com/cutensor";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cusparselt = mkWheelPkg {
    pname = "nvidia-cusparselt";
    wheelInfo = wheels.cusparselt;
    meta = {
      description = "NVIDIA cuSPARSELt ${wheels.cusparselt.version} (from PyPI)";
      homepage = "https://developer.nvidia.com/cusparselt";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };
}

# ══════════════════════════════════════════════════════════════════════════════
# CUTLASS (header-only from GitHub)
# ══════════════════════════════════════════════════════════════════════════════

// {
  nvidia-cutlass = prev.stdenv.mkDerivation {
    pname = "nvidia-cutlass";
    version = "4.3.3";

    src = fetchurl {
      url = "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.3.3.zip";
      hash = "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w=";
    };

    nativeBuildInputs = [ final.unzip ];

    dontConfigure = true;
    dontBuild = true;

    installPhase = ''
      runHook preInstall
      mkdir -p $out/include
      cp -r include/* $out/include/
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA CUTLASS 4.3.3 (header-only)";
      homepage = "https://github.com/NVIDIA/cutlass";
      license = lib.licenses.bsd3;
      platforms = lib.platforms.all;
    };
  };
}

# ══════════════════════════════════════════════════════════════════════════════
# Container-based packages (for full CUDA toolkit with nvcc, etc.)
# ══════════════════════════════════════════════════════════════════════════════

// lib.optionalAttrs (tritonRootfs != null) {
  # Full CUDA SDK from container (toolkit binaries like nvcc)
  nvidia-cuda-toolkit = mkContainerPkg {
    pname = "nvidia-cuda-toolkit";
    version = "13.0.1";
    rootfs = tritonRootfs;
    extractMode = "cuda";
    runtimeInputs = extendedRuntimeDeps;
    meta = {
      description = "NVIDIA CUDA Toolkit 13.0.1 (from NGC container)";
      license = lib.licenses.unfree;
      platforms = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    };
  };

  # Triton Inference Server
  nvidia-tritonserver = mkContainerPkg {
    pname = "nvidia-tritonserver";
    version = "25.11";
    rootfs = tritonRootfs;
    extractMode = "triton";
    runtimeInputs = tritonRuntimeDeps;
    ignoreMissingDeps = [
      # LLVM/Clang
      "libLLVM.so.18.1"
      # GC/ObjC
      "libgc.so.1"
      "libobjc_gc.so.4.0.0"
      # Misc utilities
      "libonig.so.5"
      "libmpfr.so.6"
      "libxxhash.so.0"
      "libjq.so.1.0.4"
      "libsasl2.so.2"
      "libapt-pkg.so.6.0"
      "libapt-private.so.0.0"
      # CUDA libs (provided at runtime via LD_LIBRARY_PATH)
      "libcaffe2_nvrtc.so"
      "libcufile.so.0"
      "libOpenCL.so.1"
      # CUDA 12.x compat (backends compiled against older CUDA)
      "libcusolver.so.12"
      "libcusparse.so.12"
      "libcufft.so.12"
      "libcurand.so.10"
    ];
    # Symlinks are now created by Haskell script (createLibrarySymlinks)
    postExtract = ''
      # Fix Python shebangs
      find $out -type f \( -name "*.py" -o -perm /u+x \) 2>/dev/null | while read -r f; do
        if [ -f "$f" ] && head -1 "$f" 2>/dev/null | grep -q '^#!.*python'; then
          sed -i "1s|^#!.*python.*|#!${final.python312}/bin/python|" "$f" 2>/dev/null || true
        fi
      done
    '';
    meta = {
      description = "NVIDIA Triton Inference Server 25.11";
      homepage = "https://developer.nvidia.com/nvidia-triton-inference-server";
      license = lib.licenses.bsd3;
      platforms = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      mainProgram = "tritonserver";
    };
  };
}
