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
_final: prev:
let
  # Access haskell.lib functions via attribute access (linter allows camelCase in attrpaths)
  # :: t12
  # :: t13
  # :: t14
  # :: t15
  # :: t16
  haskell-lib = prev.haskell.lib;
  do-jailbreak = haskell-lib.doJailbreak;
  dont-check = haskell-lib.dontCheck;
  # :: t17
  # :: String
  append-patch = haskell-lib.appendPatch;
  # :: Path
  add-build-depends = haskell-lib.addBuildDepends;

  # :: t91
  # :: t19 -> t20 -> { cryptonite : t84, ghc-source-gen : t25, grapesy : t84, hasktorch : t82, hnix : t90, hnix-store-core : t90, hnix-store-remote : t90, http2 : t52, http2-tls : t58, libtorch-ffi : t84, libtorch-ffi-helper : t90, network-run : t55, proto-lens : t90, proto-lens-protobuf-types : t90, proto-lens-protoc : t90, proto-lens-runtime : t90, proto-lens-setup : t35, tls : t61, tree-sitter : t90, tree-sitter-haskell : t90, tree-sitter-python : t90, tree-sitter-rust : t90, tree-sitter-tsx : t90, tree-sitter-typescript : t90 }
  # CUDA libraries needed for libtorch at runtime
  # Use nvidia-sdk (CUDA 13.0) which has SONAME 12 matching libtorch 2.9.0
  nvidia-sdk = prev.nvidia-sdk or (throw "nvidia-sdk not available - enable aleph.nixpkgs.nv");
  cuda-lib-path = "${nvidia-sdk}/lib";
  # :: t25
  # Patch for proto-lens-setup to fix Cabal 3.14+ SymbolicPath API changes
  proto-lens-setup-patch = ./patches/proto-lens-setup-cabal-3.14.patch;

  # GHC 9.12 package set with overrides
  hs-pkgs = prev.haskell.packages.ghc912.override {
    overrides = hself: hsuper: {
      # :: t27
      # :: t29
      # :: t31
      # :: t35
      # :: t37
      # ────────────────────────────────────────────────────────────────────────
      # ghc-source-gen from git (Hackage 0.4.6.0 doesn't support GHC 9.12)
      # Required by: proto-lens-protoc -> grapesy
      # ────────────────────────────────────────────────────────────────────────
      # :: t39
      # :: t41
      # :: t43
      # :: t45
      # :: t47
      # :: t49
      ghc-source-gen = hself.callCabal2nix "ghc-source-gen" inputs.ghc-source-gen-src { };

      # ────────────────────────────────────────────────────────────────────────
      # proto-lens stack - needs:
      # :: t52
      # :: "http2"
      # :: "5.3.9"
      # :: "sha256-SL34bd00BWc6MK+Js6LbNdavX3o/Xce180v/HLz5n6Y="
      #   1. jailbreak for GHC 9.12 (base 4.21, ghc-prim 0.13)
      #   2. patch for Cabal 3.14+ SymbolicPath API (proto-lens-setup only)
      # :: t55
      # :: "network-run"
      # :: "0.4.3"
      # :: "sha256-MYsziRQsK6kDWE+tMIv+tIl3K/BHw5ATFkNoPnss7CQ="
      # ────────────────────────────────────────────────────────────────────────
      proto-lens = do-jailbreak hsuper.proto-lens;
      # :: t58
      # :: "http2-tls"
      # :: "0.4.5"
      # :: "sha256-pvbRUBHs4AvpVL4qOKJjIdfIuBxU8C84OyroW4fPF2w="
      proto-lens-runtime = do-jailbreak hsuper.proto-lens-runtime;
      proto-lens-protoc = do-jailbreak hsuper.proto-lens-protoc;
      # :: t61
      # :: "tls"
      # :: "2.1.4"
      # :: "sha256-IhfECyq50ipDvbAMhNuhmLu5F6lLYH8q+/jotcPlUog="
      proto-lens-setup = append-patch (do-jailbreak hsuper.proto-lens-setup) proto-lens-setup-patch;
      proto-lens-protobuf-types = do-jailbreak hsuper.proto-lens-protobuf-types;
# :: t65

      # :: "grapesy"
      # :: "1.0.0"
      # :: "sha256-oD2+Td4eKJyDNu1enFf91Mmi4hvh0QFrJluYw9IfnvA="
      # ────────────────────────────────────────────────────────────────────────
      # tree-sitter stack - jailbreak for GHC 9.12 (containers/filepath bounds)
      # ────────────────────────────────────────────────────────────────────────
      tree-sitter = do-jailbreak hsuper.tree-sitter;
      tree-sitter-python = do-jailbreak hsuper.tree-sitter-python;
      tree-sitter-typescript = do-jailbreak hsuper.tree-sitter-typescript;
      tree-sitter-tsx = do-jailbreak hsuper.tree-sitter-tsx;
      tree-sitter-haskell = do-jailbreak hsuper.tree-sitter-haskell;
      tree-sitter-rust = do-jailbreak hsuper.tree-sitter-rust;

      # ────────────────────────────────────────────────────────────────────────
      # grapesy stack - specific versions required for compatibility
      # ────────────────────────────────────────────────────────────────────────
      http2 = hself.callHackageDirect {
        pkg = "http2";
        # :: t67
        ver = "5.3.9";
        # :: t74
        sha256 = "sha256-SL34bd00BWc6MK+Js6LbNdavX3o/Xce180v/HLz5n6Y=";
      # :: t71
      # :: t73
      } { };

      network-run = hself.callHackageDirect {
        # :: t82
        pkg = "network-run";
        ver = "0.4.3";
        # :: String
        sha256 = "sha256-MYsziRQsK6kDWE+tMIv+tIl3K/BHw5ATFkNoPnss7CQ=";
      } { };

      http2-tls = hself.callHackageDirect {
        pkg = "http2-tls";
        ver = "0.4.5";
        # :: t84
        # :: t86
        # :: t88
        # :: t90
        sha256 = "sha256-pvbRUBHs4AvpVL4qOKJjIdfIuBxU8C84OyroW4fPF2w=";
      } { };

      tls = hself.callHackageDirect {
        # :: { packages : { ghc912 : t91 } }
        # :: { ghc912 : t91 }
        # :: t91
        pkg = "tls";
        ver = "2.1.4";
        sha256 = "sha256-IhfECyq50ipDvbAMhNuhmLu5F6lLYH8q+/jotcPlUog=";
      } { };

      grapesy = dont-check (
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
      libtorch-ffi-helper = do-jailbreak hsuper.libtorch-ffi-helper;

      libtorch-ffi =
        let
          base = do-jailbreak hsuper.libtorch-ffi;
          with-deps = add-build-depends base [ nvidia-sdk ];
        in
        dont-check with-deps;

      hasktorch =
        (dont-check (do-jailbreak (add-build-depends hsuper.hasktorch [ nvidia-sdk ]))).overrideAttrs
          (_old: {
            LD_LIBRARY_PATH = cuda-lib-path;
          });

      # ────────────────────────────────────────────────────────────────────────
      # hnix stack - for render.nix Nix expression parsing
      # cryptonite has a flaky test, skip it
      # ────────────────────────────────────────────────────────────────────────
      cryptonite = dont-check hsuper.cryptonite;
      hnix-store-core = do-jailbreak hsuper.hnix-store-core;
      hnix-store-remote = do-jailbreak hsuper.hnix-store-remote;
      hnix = do-jailbreak hsuper.hnix;
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
