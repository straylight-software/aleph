# overlays/nixpkgs-nvidia-sdk.nix — NVIDIA SDK from nixpkgs cudaPackages_13
#
# Uses redistributable CUDA packages from nixpkgs, not the full installer.
# For the full SDK with non-redistributables, see libmodern-nvidia-sdk.
#
_final: prev:
let
  inherit (prev) lib;

  cuda-packages = prev.cudaPackages_13;

  # ════════════════════════════════════════════════════════════════════════════
  # C++ LIBRARIES
  # ════════════════════════════════════════════════════════════════════════════

  # mdspan - C++23 std::mdspan reference implementation (Kokkos)
  mdspan = prev.stdenv.mkDerivation (finalAttrs: {
    pname = "mdspan";
    version = "0.6.0";

    src = prev.fetchFromGitHub {
      owner = "kokkos";
      repo = "mdspan";
      rev = "mdspan-${finalAttrs.version}";
      hash = "sha256-bwE+NO/n9XsWOp3GjgLHz3s0JR0CzNDernfLHVqU9Z8=";
    };

    nativeBuildInputs = [ prev.cmake ];

    cmakeFlags = [
      "-DMDSPAN_ENABLE_TESTS=OFF"
      "-DMDSPAN_ENABLE_BENCHMARKS=OFF"
      "-DMDSPAN_ENABLE_EXAMPLES=OFF"
    ];

    # Add C++23 <mdspan> shim that includes the experimental implementation
    postInstall = ''
      install -m644 ${./packages/mdspan-shim.hpp} $out/include/mdspan
    '';

    meta = {
      description = "C++23 std::mdspan reference implementation";
      homepage = "https://github.com/kokkos/mdspan";
      license = lib.licenses.asl20;
    };
  });

  # CUTLASS - latest version, header-only
  cutlass = prev.stdenv.mkDerivation (finalAttrs: {
    pname = "cutlass";
    version = "4.3.3";

    src = prev.fetchFromGitHub {
      owner = "NVIDIA";
      repo = "cutlass";
      rev = "v${finalAttrs.version}";
      hash = "sha256-uOfSEjbwn/edHEgBikC9wAarn6c6T71ebPg74rv2qlw=";
    };

    dontBuild = true;
    dontConfigure = true;

    installPhase = ''
      runHook preInstall
      mkdir -p $out/include
      cp -r include/cutlass $out/include/
      cp -r include/cute $out/include/
      runHook postInstall
    '';

    meta = {
      description = "CUDA Templates for Linear Algebra Subroutines";
      homepage = "https://github.com/NVIDIA/cutlass";
      license = lib.licenses.bsd3;
    };
  });

  # ════════════════════════════════════════════════════════════════════════════
  # NVIDIA SDK
  # ════════════════════════════════════════════════════════════════════════════

  nvidia-sdk = prev.symlinkJoin {
    name = "nvidia-sdk";

    paths = [
      # Core toolkit
      cuda-packages.cudatoolkit
      cuda-packages.cuda_cudart
      cuda-packages.cuda_nvcc
      cuda-packages.cuda_nvrtc
      cuda-packages.cuda_cupti
      cuda-packages.cuda_gdb
      cuda-packages.cuda_sanitizer_api
      cuda-packages.cuda_cccl

      # Math
      cuda-packages.libcublas
      cuda-packages.libcufft
      cuda-packages.libcurand
      cuda-packages.libcusolver
      cuda-packages.libcusparse
      cuda-packages.libnvjitlink

      # ML
      cuda-packages.cudnn
      cuda-packages.tensorrt
      cuda-packages.nccl

      # CUTLASS
      cutlass
    ];

    postBuild = builtins.readFile ./scripts/nvidia-sdk-postbuild.sh;

    passthru = {
      inherit cuda-packages cutlass;
      inherit (cuda-packages.cudatoolkit) version;
    };

    meta = {
      description = "NVIDIA SDK (cuda-packages_13)";
      license = lib.licenses.unfree;
      platforms = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    };
  };

in
lib.optionalAttrs prev.stdenv.isLinux {
  inherit nvidia-sdk cutlass cuda-packages;
}
// {
  inherit mdspan cutlass;
}
