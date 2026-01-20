{
  description = "FREESIDEフリーサイド — WHY WAIT?";

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

    # Buck2 prelude (straylight fork with NVIDIA support)
    # Mercury-based Haskell rules, LLVM 22 C++ toolchain, nv target compilation
    buck2-prelude = {
      url = "github:weyl-ai/straylight-buck2-prelude";
      flake = false;
    };
  };

  outputs =
    inputs@{ flake-parts, ... }: flake-parts.lib.mkFlake { inherit inputs; } (import ./nix/_main.nix);
}
