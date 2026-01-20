# nix/overlays/ghc-wasm.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // ghc-wasm //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The sky above the port was the color of television, tuned to a
#     dead channel.
#
#                                                         — Neuromancer
#
# GHC WASM backend toolchain from ghc-wasm-meta.
# Provides wasm32-wasi-ghc, wasm32-wasi-cabal, and related tools.
#
# This enables:
#   - Compiling Haskell to WASM for use with straylight-nix's builtins.wasm
#   - Type-safe derivation builders in Haskell
#   - Pure Haskell evaluation during Nix builds
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
let
  # ghc-wasm-meta provides packages per-system, not as an overlay
  # We wrap it to provide a consistent interface
  mkGhcWasmPackages =
    system:
    let
      ghcWasmPkgs = inputs.ghc-wasm-meta.packages.${system} or { };
    in
    {
      # GHC 9.12 - recommended for Hackage compatibility
      ghc-wasm = ghcWasmPkgs.wasm32-wasi-ghc-9_12 or null;
      ghc-wasm-cabal = ghcWasmPkgs.wasm32-wasi-cabal-9_12 or null;

      # Alternative GHC versions
      ghc-wasm-9_14 = ghcWasmPkgs.wasm32-wasi-ghc-9_14 or null;
      ghc-wasm-9_10 = ghcWasmPkgs.wasm32-wasi-ghc-9_10 or null;
      ghc-wasm-9_8 = ghcWasmPkgs.wasm32-wasi-ghc-9_8 or null;

      # GMP flavour (default, uses libgmp)
      ghc-wasm-gmp = ghcWasmPkgs.wasm32-wasi-ghc-gmp or null;

      # Native flavour (uses native Haskell Integer)
      ghc-wasm-native = ghcWasmPkgs.wasm32-wasi-ghc-native or null;

      # Supporting tools
      wasi-sdk = ghcWasmPkgs.wasi-sdk or null;
      wasm-binaryen = ghcWasmPkgs.binaryen or null;
      wasm-wasmtime = ghcWasmPkgs.wasmtime or null;

      # All-in-one bundles
      ghc-wasm-all = ghcWasmPkgs.all_9_12 or null;
      ghc-wasm-all-9_14 = ghcWasmPkgs.all_9_14 or null;
    };
in
{
  flake.overlays.ghc-wasm = final: prev: {
    straylight = (prev.straylight or { }) // {
      ghc-wasm = mkGhcWasmPackages final.stdenv.hostPlatform.system;
    };
  };
}
