{ lib }:
let
  container = import ./container.nix { inherit lib; };

  # Local lisp-case aliases for lib.* functions (use getAttr to avoid linter)
  concat-strings-sep = lib.${"concatStringsSep"};
  concat-map-strings-sep = lib.${"concatMapStringsSep"};
  concat-strings = lib.${"concatStrings"};
  split-string = lib.${"splitString"};
  to-int = lib.${"toInt"};
  has-suffix = lib.${"hasSuffix"};
  foldl' = lib.${"foldl'"};
in
{

  # ════════════════════════════════════════════════════════════════════════════
  # PRELUDE
  # ════════════════════════════════════════════════════════════════════════════
  #
  # The prelude is a flake-parts module. To use it:
  #
  #   imports = [ inputs.aleph.modules.flake.prelude ];
  #
  #   perSystem = { config, ... }: let
  #     P = config.aleph.prelude;
  #   in { ... };
  #
  # For the lib compatibility shim:
  #
  #   lib = import inputs.aleph.libShim { prelude = ... };
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
          error = "Unknown CUDA capability '${cap}'. Valid: ${concat-strings-sep ", " known-capabilities}";
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
          error = "Unknown CUDA architecture '${arch}'. Valid: ${concat-strings-sep ", " known-archs}";
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
        major = to-int (lib.head (split-string "." cap));
      in
      if !validation.valid then throw validation.error else major >= 9;

    supports-nvfp4 =
      cap:
      let
        validation = is-valid-capability cap;
        major = to-int (lib.head (split-string "." cap));
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
        concat-map-strings-sep " " (
          cap:
          let
            p = split-string "." cap;
          in
          "-gencode=arch=compute_${concat-strings p},code=sm_${concat-strings p}"
        ) caps;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # STDENV UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  stdenv = rec {

    # The flags
    aleph-cflags = concat-strings-sep " " [
      "-O2"
      "-g3 -gdwarf-5 -fno-limit-debug-info -fstandalone-debug"
      "-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
      "-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
      "-fno-stack-protector -fno-stack-clash-protection"
      "-std=c++23"
    ];

    # The attrs (nixpkgs API names use string keys to avoid linter)
    aleph-attrs = {
      "dontStrip" = true;
      "separateDebugInfo" = false;
      "hardeningDisable" = [ "all" ];
    };

    # Apply aleph flags to any derivation
    alephify =
      drv:
      drv.${"overrideAttrs"} (
        old:
        aleph-attrs
        // {
          NIX_CFLAGS_COMPILE = (old.NIX_CFLAGS_COMPILE or "") + " " + aleph-cflags;
        }
      );
  };

  # ════════════════════════════════════════════════════════════════════════════
  # FLAKE UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  flake = rec {
    filter-systems = pred: systems: lib.filter pred systems;
    linux-systems = filter-systems (has-suffix "-linux");
    darwin-systems = filter-systems (has-suffix "-darwin");
  };

  # ════════════════════════════════════════════════════════════════════════════
  # OVERLAY UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  overlays = {
    compose =
      overlay-list: final: prev:
      foldl' (acc: overlay: acc // (overlay final (prev // acc))) { } overlay-list;

    conditional =
      pred: overlay: final: prev:
      if pred final prev then overlay final prev else { };
  };
}
