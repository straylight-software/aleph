# nvidia-sdk overlay
#
# Container rootfs fetching for NVIDIA SDK extraction.
# Extraction logic is in Aleph.Script.Nvidia.Container (Haskell).
#
# Architecture:
#   1. container-to-nix: FOD that pulls container image (cached, network access)
#   2. packages.nix: Uses nvidia-sdk Haskell script to extract from rootfs
#
# Usage:
#   pkgs.nvidia-sdk-ngc-rootfs   # Raw rootfs for debugging
#   pkgs.nvidia-cuda-toolkit     # CUDA toolkit (via packages.nix)
#   pkgs.nvidia-tritonserver     # Triton server (via packages.nix)
#
final: prev:
let
  inherit (prev) lib;
  inherit (prev.stdenv.hostPlatform) system;

  # Import prelude for translate-attrs
  translations = import ../../prelude/translations.nix { inherit lib; };
  inherit (translations) translate-attrs;

  # ════════════════════════════════════════════════════════════════════════════
  # container-to-nix: Fixed-output derivation for container pulling
  # ════════════════════════════════════════════════════════════════════════════
  #
  # This is a FOD - Nix allows network access and verifies the hash.
  # The output is the unpacked container rootfs.

  container-to-nix =
    {
      name,
      image-ref,
      hash,
    }:
    prev.stdenvNoCC.mkDerivation (
      translate-attrs {
        inherit name;

        native-build-inputs = [
          final.crane
          final.gnutar
          final.gzip
        ];

        # SSL certs for HTTPS
        SSL_CERT_FILE = "${final.cacert}/etc/ssl/certs/ca-bundle.crt";

        meta = {
          description = "NVIDIA container image rootfs for SDK extraction";
        };
      }
      // {
        # NOTE: FOD attrs are nixpkgs API, quoted
        "outputHashAlgo" = "sha256";
        "outputHashMode" = "recursive";
        "outputHash" = hash;

        "buildCommand" = ''
          mkdir -p $out
          crane export ${image-ref} - | tar -xf - -C $out
        '';
      }
    );

  # ════════════════════════════════════════════════════════════════════════════
  # Container definitions
  # ════════════════════════════════════════════════════════════════════════════

  containers = {
    tritonserver = {
      version = "25.11";
      "x86_64-linux" = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3";
        hash = "sha256-yrTbMURSSc5kx4KTegTErpDjCWcjb9Ehp7pOUtP34pM=";
      };
      "aarch64-linux" = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3-igpu";
        hash = ""; # Not yet computed
      };
    };

    cuda-devel = {
      version = "13.0.1";
      "x86_64-linux" = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = ""; # Not yet computed
      };
      "aarch64-linux" = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = "";
      };
    };
  };

  # Container info for current system
  has-triton-info = containers.tritonserver ? ${system};
  has-cuda-info = containers.cuda-devel ? ${system};

  triton-info = containers.tritonserver.${system};
  cuda-info = containers.cuda-devel.${system};

  # Container rootfs FODs (only defined if hash is provided)
  has-triton-rootfs = has-triton-info && triton-info.hash != "";
  has-cuda-rootfs = has-cuda-info && cuda-info.hash != "";

  triton-rootfs = container-to-nix {
    name = "tritonserver-${containers.tritonserver.version}-rootfs";
    image-ref = triton-info.ref;
    inherit (triton-info) hash;
  };

  cuda-rootfs = container-to-nix {
    name = "cuda-devel-${containers.cuda-devel.version}-rootfs";
    image-ref = cuda-info.ref;
    inherit (cuda-info) hash;
  };

in
lib.optionalAttrs has-triton-rootfs {
  # Expose rootfs for packages.nix and debugging
  nvidia-sdk-ngc-rootfs = triton-rootfs;
}
// lib.optionalAttrs has-cuda-rootfs {
  nvidia-sdk-cuda-rootfs = cuda-rootfs;
}
// {
  # Container definitions for reference
  nvidia-sdk-containers = containers;
}
