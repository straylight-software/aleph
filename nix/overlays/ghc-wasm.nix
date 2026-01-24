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
  # Check if ghc-wasm-meta input is available
  hasGhcWasmMeta = inputs ? ghc-wasm-meta;

  # ghc-wasm-meta provides packages per-system, not as an overlay
  # We wrap it to provide a consistent interface
  mkGhcWasmPackages =
    system:
    let
      ghcWasmPkgs = inputs.ghc-wasm-meta.packages.${system};

      # Helper to conditionally include an attribute
      optionalPkg = name: if ghcWasmPkgs ? ${name} then { ${name} = ghcWasmPkgs.${name}; } else { };
    in
    # Core packages (must exist)
    {
      ghc-wasm = ghcWasmPkgs.wasm32-wasi-ghc-9_12;
      ghc-wasm-cabal = ghcWasmPkgs.wasm32-wasi-cabal-9_12;
      wasi-sdk = ghcWasmPkgs.wasi-sdk;
    }
    # Alternative GHC versions (optional)
    // optionalPkg "wasm32-wasi-ghc-9_14"
    // optionalPkg "wasm32-wasi-ghc-9_10"
    // optionalPkg "wasm32-wasi-ghc-9_8"
    // optionalPkg "wasm32-wasi-ghc-gmp"
    // optionalPkg "wasm32-wasi-ghc-native"
    // optionalPkg "binaryen"
    // optionalPkg "wasmtime"
    // optionalPkg "all_9_12"
    // optionalPkg "all_9_14"
    # Rename to our naming convention
    // (
      if ghcWasmPkgs ? wasm32-wasi-ghc-9_14 then
        { ghc-wasm-9_14 = ghcWasmPkgs.wasm32-wasi-ghc-9_14; }
      else
        { }
    )
    // (
      if ghcWasmPkgs ? wasm32-wasi-ghc-9_10 then
        { ghc-wasm-9_10 = ghcWasmPkgs.wasm32-wasi-ghc-9_10; }
      else
        { }
    )
    // (
      if ghcWasmPkgs ? wasm32-wasi-ghc-9_8 then
        { ghc-wasm-9_8 = ghcWasmPkgs.wasm32-wasi-ghc-9_8; }
      else
        { }
    )
    // (
      if ghcWasmPkgs ? wasm32-wasi-ghc-gmp then
        { ghc-wasm-gmp = ghcWasmPkgs.wasm32-wasi-ghc-gmp; }
      else
        { }
    )
    // (
      if ghcWasmPkgs ? wasm32-wasi-ghc-native then
        { ghc-wasm-native = ghcWasmPkgs.wasm32-wasi-ghc-native; }
      else
        { }
    )
    // (if ghcWasmPkgs ? binaryen then { wasm-binaryen = ghcWasmPkgs.binaryen; } else { })
    // (if ghcWasmPkgs ? wasmtime then { wasm-wasmtime = ghcWasmPkgs.wasmtime; } else { })
    // (if ghcWasmPkgs ? all_9_12 then { ghc-wasm-all = ghcWasmPkgs.all_9_12; } else { })
    // (if ghcWasmPkgs ? all_9_14 then { ghc-wasm-all-9_14 = ghcWasmPkgs.all_9_14; } else { });
in
{
  flake.overlays.ghc-wasm =
    final: prev:
    if hasGhcWasmMeta then
      {
        straylight = (prev.straylight or { }) // {
          ghc-wasm = mkGhcWasmPackages final.stdenv.hostPlatform.system;
        };
      }
    else
      { };
}
