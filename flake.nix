{
  description = "// straylight // aleph // continuity project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    ndg.url = "github:feel-co/ndg";

    llvm-project = {
      url = "github:llvm/llvm-project/bb1f220d534b0f6d80bea36662f5188ff11c2e54";
      flake = false;
    };

    # Determinate Nix with WASM support (builtins.wasm + wasm32-wasip1)
    # NOTE: Don't follow nixpkgs - needs specific rust version for wasmtime
    nix.url = "github:straylight-software/nix";

    # GHC WASM backend toolchain (ghc-wasm-meta)
    # Provides wasm32-wasi-ghc, wasm32-wasi-cabal, wasi-sdk, wasmtime, etc.
    # Using GitHub mirror since gitlab.haskell.org has different flake URL format
    ghc-wasm-meta = {
      url = "github:haskell-wasm/ghc-wasm-meta";
      # Don't follow nixpkgs - ghc-wasm-meta has specific version requirements
    };

    # ghc-source-gen from git (Hackage version doesn't support GHC 9.12)
    # Required for grapesy -> proto-lens-protoc -> ghc-source-gen
    ghc-source-gen-src = {
      url = "github:google/ghc-source-gen";
      flake = false;
    };

    # Buck2 prelude (straylight fork with NVIDIA support)
    # Mercury-based Haskell rules, LLVM 22 C++ toolchain, nv target compilation
    buck2-prelude = {
      url = "github:weyl-ai/straylight-buck2-prelude";
      flake = false;
    };

    # NativeLink - Local Remote Execution for Buck2
    # Provides CAS, scheduler, and worker for build caching
    nativelink.url = "github:TraceMachina/nativelink";

    # nix2gpu - NixOS containers for GPU compute (vast.ai, runpod, fly.io)
    # Uses nimi as PID 1 for modular services
    nix2gpu = {
      url = "github:fleek-sh/nix2gpu";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Nimi - Tini-like PID 1 for containers and NixOS modular services
    # Used for VM init scripts with proper process supervision
    # Using fork with nimi-static for microVM support (PR: weyl-ai/nimi#12)
    nimi = {
      url = "github:b7r6/nimi/add-static-build";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }: flake-parts.lib.mkFlake { inherit inputs; } (import ./nix/_main.nix);
}
