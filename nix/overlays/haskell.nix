# nix/overlays/haskell.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // haskell //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Haskell package overrides for GHC 9.12:
#   - ghc-source-gen from git (required for grapesy, not on Hackage for 9.12)
#   - grapesy and dependencies with correct versions
#   - proto-lens stack patched for Cabal 3.14+ SymbolicPath API
#
# This overlay modifies haskell.packages.ghc912 which is used by the build
# module (toolchains.nix) for Buck2 Haskell compilation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
final: prev:
let
  inherit (prev.haskell.lib)
    doJailbreak
    dontCheck
    appendPatch
    addBuildDepends
    overrideCabal
    ;

  # CUDA libraries needed for libtorch at runtime
  # Use nvidia-sdk (CUDA 13.0) which has SONAME 12 matching libtorch 2.9.0
  nvidia-sdk = prev.nvidia-sdk or (throw "nvidia-sdk not available - enable aleph.nixpkgs.nv");
  cuda-lib-path = "${nvidia-sdk}/lib";
  # Patch for proto-lens-setup to fix Cabal 3.14+ SymbolicPath API changes
  proto-lens-setup-patch = ./patches/proto-lens-setup-cabal-3.14.patch;

  # GHC 9.12 package set with overrides
  hs-pkgs = prev.haskell.packages.ghc912.override {
    overrides = hself: hsuper: {
      # ────────────────────────────────────────────────────────────────────────
      # ghc-source-gen from git (Hackage 0.4.6.0 doesn't support GHC 9.12)
      # Required by: proto-lens-protoc -> grapesy
      # ────────────────────────────────────────────────────────────────────────
      ghc-source-gen = hself.callCabal2nix "ghc-source-gen" inputs.ghc-source-gen-src { };

      # ────────────────────────────────────────────────────────────────────────
      # proto-lens stack - needs:
      #   1. jailbreak for GHC 9.12 (base 4.21, ghc-prim 0.13)
      #   2. patch for Cabal 3.14+ SymbolicPath API (proto-lens-setup only)
      # ────────────────────────────────────────────────────────────────────────
      proto-lens = doJailbreak hsuper.proto-lens;
      proto-lens-runtime = doJailbreak hsuper.proto-lens-runtime;
      proto-lens-protoc = doJailbreak hsuper.proto-lens-protoc;
      proto-lens-setup = appendPatch (doJailbreak hsuper.proto-lens-setup) proto-lens-setup-patch;
      proto-lens-protobuf-types = doJailbreak hsuper.proto-lens-protobuf-types;

      # ────────────────────────────────────────────────────────────────────────
      # grapesy stack - specific versions required for compatibility
      # ────────────────────────────────────────────────────────────────────────
      http2 = hself.callHackageDirect {
        pkg = "http2";
        ver = "5.3.9";
        sha256 = "sha256-SL34bd00BWc6MK+Js6LbNdavX3o/Xce180v/HLz5n6Y=";
      } { };

      network-run = hself.callHackageDirect {
        pkg = "network-run";
        ver = "0.4.3";
        sha256 = "sha256-MYsziRQsK6kDWE+tMIv+tIl3K/BHw5ATFkNoPnss7CQ=";
      } { };

      http2-tls = hself.callHackageDirect {
        pkg = "http2-tls";
        ver = "0.4.5";
        sha256 = "sha256-pvbRUBHs4AvpVL4qOKJjIdfIuBxU8C84OyroW4fPF2w=";
      } { };

      tls = hself.callHackageDirect {
        pkg = "tls";
        ver = "2.1.4";
        sha256 = "sha256-IhfECyq50ipDvbAMhNuhmLu5F6lLYH8q+/jotcPlUog=";
      } { };

      grapesy = dontCheck (
        hself.callHackageDirect {
          pkg = "grapesy";
          ver = "1.0.0";
          sha256 = "sha256-oD2+Td4eKJyDNu1enFf91Mmi4hvh0QFrJluYw9IfnvA=";
        } { }
      );

      # ────────────────────────────────────────────────────────────────────────
      # Hasktorch - typed tensor bindings to libtorch
      #
      # libtorch-ffi-helper: Has ghc <9.12 constraint, jailbreak to allow 9.12.
      # libtorch-ffi/hasktorch: Need nvidia-sdk (CUDA 13.0) because nixpkgs
      #   libtorch 2.9.0 is a prebuilt binary from PyTorch built against CUDA
      #   13.0 (SONAME .so.12). nixpkgs cudaPackages_12_8 provides SONAME .so.11.
      #
      # hasktorch: GHC loads libtorch-ffi at compile time, which dlopens
      #   libtorch.so, which needs CUDA libs. We set LD_LIBRARY_PATH at the
      #   derivation level to point to nvidia-sdk/lib.
      # ────────────────────────────────────────────────────────────────────────
      libtorch-ffi-helper = doJailbreak hsuper.libtorch-ffi-helper;

      libtorch-ffi =
        let
          base = doJailbreak hsuper.libtorch-ffi;
          with-deps = addBuildDepends base [ nvidia-sdk ];
        in
        dontCheck with-deps;

      hasktorch =
        (dontCheck (doJailbreak (addBuildDepends hsuper.hasktorch [ nvidia-sdk ]))).overrideAttrs
          (old: {
            LD_LIBRARY_PATH = cuda-lib-path;
          });
    };
  };
in
{
  haskell = prev.haskell // {
    packages = prev.haskell.packages // {
      ghc912 = hs-pkgs;
    };
  };
}
