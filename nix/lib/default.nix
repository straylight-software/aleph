{ lib }:
let
  container = import ./container.nix { inherit lib; };
in
{

  # ════════════════════════════════════════════════════════════════════════════
  # PRELUDE
  # ════════════════════════════════════════════════════════════════════════════
  #
  # The prelude is a flake-parts module. To use it:
  #
  #   imports = [ inputs.aleph-naught.modules.flake.prelude ];
  #
  #   perSystem = { config, ... }: let
  #     P = config.straylight.prelude;
  #   in { ... };
  #
  # For the lib compatibility shim:
  #
  #   lib = import inputs.aleph-naught.libShim { prelude = ... };
  #
  # ════════════════════════════════════════════════════════════════════════════

  # ════════════════════════════════════════════════════════════════════════════
  # CONTAINER / NAMESPACE / FIRECRACKER UTILITIES
  # ════════════════════════════════════════════════════════════════════════════
  #
  # Pure functions for working with OCI images, Linux namespaces, and Firecracker VMs.
  # See nix/lib/container.nix for implementation.
  #
  # Usage:
  #   lib.oci.parse-ref "nvcr.io/nvidia/pytorch:25.01-py3"
  #   lib.namespace.gpu-flags
  #   lib.firecracker.mk-config { ... }
  #   lib.elf.mk-rpath [ pkgs.zlib pkgs.openssl ]
  #   lib.pep503.normalize-name "Foo_Bar"
  #
  # ════════════════════════════════════════════════════════════════════════════

  inherit (container)
    oci
    namespace
    firecracker
    elf
    pep503
    ;

  # ════════════════════════════════════════════════════════════════════════════
  # NVIDIA GPU UTILITIES
  # ════════════════════════════════════════════════════════════════════════════
  #
  # We say "nvidia" and abbreviate "nv", not "cuda".
  # See: docs/languages/nix/philosophy/nvidia-not-cuda.md
  #
  # CUDA is NVIDIA's marketing term. We're building on their hardware,
  # not adopting their identity.
  #

  nv = rec {

    # Known capabilities for validation
    known-capabilities = [
      "7.0"
      "7.5"
      "8.0"
      "8.6"
      "8.7"
      "8.9"
      "9.0"
      "10.0"
      "12.0"
      "12.1"
    ];

    # Known architectures for validation
    known-archs = [
      "volta"
      "turing"
      "ampere"
      "orin"
      "ada"
      "hopper"
      "thor"
      "blackwell"
    ];

    # Validate a capability string - returns { valid, value, error }
    is-valid-capability =
      cap:
      if builtins.elem cap known-capabilities then
        {
          valid = true;
          value = cap;
          error = null;
        }
      else
        {
          valid = false;
          value = null;
          error = "Unknown CUDA capability '${cap}'. Valid: ${lib.concatStringsSep ", " known-capabilities}";
        };

    # Validate an architecture string - returns { valid, value, error }
    is-valid-arch =
      arch:
      if builtins.elem arch known-archs then
        {
          valid = true;
          value = arch;
          error = null;
        }
      else
        {
          valid = false;
          value = null;
          error = "Unknown CUDA architecture '${arch}'. Valid: ${lib.concatStringsSep ", " known-archs}";
        };

    capability-to-arch =
      cap:
      let
        validation = is-valid-capability cap;
      in
      if !validation.valid then
        throw validation.error
      else
        {
          "7.0" = "volta";
          "7.5" = "turing";
          "8.0" = "ampere";
          "8.6" = "ampere";
          "8.7" = "orin";
          "8.9" = "ada";
          "9.0" = "hopper";
          "10.0" = "blackwell"; # 2CTA
          "12.0" = "blackwell"; # 1CTA
          "12.1" = "blackwell"; # 1CTA / grace
        }
        .${cap};

    arch-to-capability =
      arch:
      let
        validation = is-valid-arch arch;
      in
      if !validation.valid then
        throw validation.error
      else
        {
          "volta" = "7.0";
          "turing" = "7.5";
          "ampere" = "8.0";
          "orin" = "8.7";
          "ada" = "8.9";
          "hopper" = "9.0";
          "thor" = "9.0";
          "blackwell" = "12.0";
        }
        .${arch};

    supports-fp8 =
      cap:
      let
        validation = is-valid-capability cap;
        major = lib.toInt (lib.head (lib.splitString "." cap));
      in
      if !validation.valid then throw validation.error else major >= 9;

    supports-nvfp4 =
      cap:
      let
        validation = is-valid-capability cap;
        major = lib.toInt (lib.head (lib.splitString "." cap));
      in
      if !validation.valid then throw validation.error else major >= 12;

    nvcc-flags =
      caps:
      let
        validations = map is-valid-capability caps;
        errors = lib.filter (v: !v.valid) validations;
      in
      if errors != [ ] then
        throw (lib.head errors).error
      else
        lib.concatMapStringsSep " " (
          cap:
          let
            p = lib.splitString "." cap;
          in
          "-gencode=arch=compute_${lib.concatStrings p},code=sm_${lib.concatStrings p}"
        ) caps;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # STDENV UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  stdenv = rec {

    # The flags
    straylight-cflags = lib.concatStringsSep " " [
      "-O2"
      "-g3 -gdwarf-5 -fno-limit-debug-info -fstandalone-debug"
      "-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
      "-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
      "-fno-stack-protector -fno-stack-clash-protection"
      "-std=c++23"
    ];

    # The attrs (nixpkgs API names are kept as-is)
    straylight-attrs = {
      dontStrip = true;
      separateDebugInfo = false;
      hardeningDisable = [ "all" ];
    };

    # Apply straylight flags to any derivation
    straylightify =
      drv:
      drv.overrideAttrs (
        old:
        straylight-attrs
        // {
          NIX_CFLAGS_COMPILE = (old.NIX_CFLAGS_COMPILE or "") + " " + straylight-cflags;
        }
      );
  };

  # ════════════════════════════════════════════════════════════════════════════
  # FLAKE UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  flake = rec {
    filter-systems = pred: systems: lib.filter pred systems;
    linux-systems = filter-systems (lib.hasSuffix "-linux");
    darwin-systems = filter-systems (lib.hasSuffix "-darwin");
  };

  # ════════════════════════════════════════════════════════════════════════════
  # OVERLAY UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  overlays = {
    compose =
      overlay-list: final: prev:
      lib.foldl' (acc: overlay: acc // (overlay final (prev // acc))) { } overlay-list;

    conditional =
      pred: overlay: final: prev:
      if pred final prev then overlay final prev else { };
  };
}
