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
  has-ghc-wasm-meta = inputs ? ghc-wasm-meta;

  # ghc-wasm-meta provides packages per-system, not as an overlay
  # We wrap it to provide a consistent interface
  mk-ghc-wasm-packages =
    system:
    let
      ghc-wasm-pkgs = inputs.ghc-wasm-meta.packages.${system};

      # Helper to conditionally include an attribute
      optional-pkg = name: if ghc-wasm-pkgs ? ${name} then { ${name} = ghc-wasm-pkgs.${name}; } else { };
    in
    # Core packages (must exist)
    {
      ghc-wasm = ghc-wasm-pkgs.wasm32-wasi-ghc-9_12;
      ghc-wasm-cabal = ghc-wasm-pkgs.wasm32-wasi-cabal-9_12;
      inherit (ghc-wasm-pkgs) wasi-sdk;
    }
    # Alternative GHC versions (optional)
    # NOTE: Package names are from ghc-wasm-meta (external API), accessed via string
    // optional-pkg "wasm32-wasi-ghc-9_14"
    // optional-pkg "wasm32-wasi-ghc-9_10"
    // optional-pkg "wasm32-wasi-ghc-9_8"
    // optional-pkg "wasm32-wasi-ghc-gmp"
    // optional-pkg "wasm32-wasi-ghc-native"
    // optional-pkg "binaryen"
    // optional-pkg "wasmtime"
    // optional-pkg "all_9_12"
    // optional-pkg "all_9_14"
    # Rename to our naming convention
    // (
      if ghc-wasm-pkgs ? "wasm32-wasi-ghc-9_14" then
        { ghc-wasm-9-14 = ghc-wasm-pkgs."wasm32-wasi-ghc-9_14"; }
      else
        { }
    )
    // (
      if ghc-wasm-pkgs ? "wasm32-wasi-ghc-9_10" then
        { ghc-wasm-9-10 = ghc-wasm-pkgs."wasm32-wasi-ghc-9_10"; }
      else
        { }
    )
    // (
      if ghc-wasm-pkgs ? "wasm32-wasi-ghc-9_8" then
        { ghc-wasm-9-8 = ghc-wasm-pkgs."wasm32-wasi-ghc-9_8"; }
      else
        { }
    )
    // (
      if ghc-wasm-pkgs ? wasm32-wasi-ghc-gmp then
        { ghc-wasm-gmp = ghc-wasm-pkgs.wasm32-wasi-ghc-gmp; }
      else
        { }
    )
    // (
      if ghc-wasm-pkgs ? wasm32-wasi-ghc-native then
        { ghc-wasm-native = ghc-wasm-pkgs.wasm32-wasi-ghc-native; }
      else
        { }
    )
    // (if ghc-wasm-pkgs ? binaryen then { wasm-binaryen = ghc-wasm-pkgs.binaryen; } else { })
    // (if ghc-wasm-pkgs ? wasmtime then { wasm-wasmtime = ghc-wasm-pkgs.wasmtime; } else { })
    // (if ghc-wasm-pkgs ? "all_9_12" then { ghc-wasm-all = ghc-wasm-pkgs."all_9_12"; } else { })
    // (if ghc-wasm-pkgs ? "all_9_14" then { ghc-wasm-all-9-14 = ghc-wasm-pkgs."all_9_14"; } else { });
in
{
  flake.overlays.ghc-wasm =
    final: prev:
    if has-ghc-wasm-meta then
      {
        aleph = (prev.aleph or { }) // {
          ghc-wasm = mk-ghc-wasm-packages final.stdenv.hostPlatform.system;
        };
      }
    else
      { };
}
