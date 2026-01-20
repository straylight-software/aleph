# nvidia-sdk overlay
#
# Container rootfs fetching for NVIDIA SDK extraction.
# Extraction logic is in Aleph.Script.Nvidia.Container (Haskell).
#
# Architecture:
#   1. containerToNix: FOD that pulls container image (cached, network access)
#   2. packages.nix: Uses nvidia-sdk Haskell script to extract from rootfs
#
# Usage:
#   pkgs.nvidia-sdk-ngc-rootfs   # Raw rootfs for debugging
#   pkgs.nvidia-cuda-toolkit     # CUDA toolkit (via packages.nix)
#   pkgs.nvidia-tritonserver     # Triton server (via packages.nix)
#
final: prev:
let
  inherit (prev) lib stdenvNoCC;
  inherit (prev.stdenv.hostPlatform) system;

  # ════════════════════════════════════════════════════════════════════════════
  # containerToNix: Fixed-output derivation for container pulling
  # ════════════════════════════════════════════════════════════════════════════
  #
  # This is a FOD - Nix allows network access and verifies the hash.
  # The output is the unpacked container rootfs.

  containerToNix =
    {
      name,
      imageRef,
      hash,
    }:
    stdenvNoCC.mkDerivation {
      inherit name;

      nativeBuildInputs = [
        final.crane
        final.gnutar
        final.gzip
      ];

      # Fixed-output derivation
      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      outputHash = hash;

      # SSL certs for HTTPS
      SSL_CERT_FILE = "${final.cacert}/etc/ssl/certs/ca-bundle.crt";

      buildCommand = ''
        mkdir -p $out
        crane export ${imageRef} - | tar -xf - -C $out
      '';
    };

  # ════════════════════════════════════════════════════════════════════════════
  # Container definitions
  # ════════════════════════════════════════════════════════════════════════════

  containers = {
    tritonserver = {
      version = "25.11";
      x86_64-linux = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3";
        hash = "sha256-yrTbMURSSc5kx4KTegTErpDjCWcjb9Ehp7pOUtP34pM=";
      };
      aarch64-linux = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3-igpu";
        hash = ""; # Not yet computed
      };
    };

    cuda-devel = {
      version = "13.0.1";
      x86_64-linux = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = ""; # Not yet computed
      };
      aarch64-linux = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = "";
      };
    };
  };

  # Container info for current system
  tritonInfo = containers.tritonserver.${system} or null;
  cudaInfo = containers.cuda-devel.${system} or null;

  # Container rootfs FODs (only defined if hash is provided)
  tritonRootfs =
    if tritonInfo != null && tritonInfo.hash != "" then
      containerToNix {
        name = "tritonserver-${containers.tritonserver.version}-rootfs";
        imageRef = tritonInfo.ref;
        inherit (tritonInfo) hash;
      }
    else
      null;

  cudaRootfs =
    if cudaInfo != null && cudaInfo.hash != "" then
      containerToNix {
        name = "cuda-devel-${containers.cuda-devel.version}-rootfs";
        imageRef = cudaInfo.ref;
        inherit (cudaInfo) hash;
      }
    else
      null;

in
lib.optionalAttrs (tritonRootfs != null) {
  # Expose rootfs for packages.nix and debugging
  nvidia-sdk-ngc-rootfs = tritonRootfs;
}
// lib.optionalAttrs (cudaRootfs != null) {
  nvidia-sdk-cuda-rootfs = cudaRootfs;
}
// {
  # Container definitions for reference
  nvidia-sdk-containers = containers;
}
