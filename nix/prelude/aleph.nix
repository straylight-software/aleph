# nix/prelude/aleph.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // aleph //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     He'd operated on an almost permanent adrenaline high, a byproduct
#     of youth and proficiency, jacked into a custom cyberspace deck
#     that projected his disembodied consciousness into the consensual
#     hallucination that was the matrix.
#
#                                                         — Neuromancer
#
# The aleph interface: evaluate typed Haskell/PureScript expressions from Nix.
#
# Zero bash. Zero untyped strings. One interface.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# USAGE:
#
#   # Basic: evaluate a typed package
#   aleph.eval "Aleph.Packages.Nvidia.nccl" {}
#
#   # With arguments
#   aleph.eval "Aleph.Build.withFlags" { pkg = myPkg; flags = ["-O3"]; }
#
#   # Import a whole module
#   nvidia = aleph.import "Aleph.Packages.Nvidia"
#   nvidia.nccl   # → derivation
#   nvidia.cudnn  # → derivation
#
# SETUP:
#
#   # In your flake:
#   wasm-infra = import ./nix/prelude/wasm-plugin.nix { ... };
#   aleph = import ./nix/prelude/aleph.nix {
#     inherit lib pkgs;
#     wasmFile = wasm-infra.alephWasm;
#   };
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  pkgs,
  wasmFile,
  stdenv-fn ? pkgs.stdenv.mkDerivation,
}:
let
  # Import the WASM plugin infrastructure
  wasm-plugin = import ./wasm-plugin.nix {
    inherit lib;
    inherit (pkgs) stdenv runCommand;
    # ghc-wasm-meta not needed for loading, only for building
    ghc-wasm-meta = null;
  };

  # ────────────────────────────────────────────────────────────────────────────
  # Module name → WASM export name mapping
  # ────────────────────────────────────────────────────────────────────────────
  # "Aleph.Packages.Nvidia.nccl" → "nvidia_nccl"
  # "Aleph.Nix.Packages.ZlibNg" → "zlib_ng"
  #
  module-to-export =
    module-path:
    let
      parts = lib.splitString "." module-path;
      # Take everything after known prefixes
      relevant-parts =
        if lib.length parts >= 3 && lib.elemAt parts 0 == "Aleph" && lib.elemAt parts 1 == "Packages" then
          lib.drop 2 parts
        else if
          lib.length parts >= 4
          && lib.elemAt parts 0 == "Aleph"
          && lib.elemAt parts 1 == "Nix"
          && lib.elemAt parts 2 == "Packages"
        then
          lib.drop 3 parts
        else
          parts;
      # CamelCase to snake_case
      to-snake =
        s:
        let
          chars = lib.stringToCharacters s;
          converted = lib.concatMapStrings (
            c:
            if lib.elem c (lib.stringToCharacters "ABCDEFGHIJKLMNOPQRSTUVWXYZ") then "_${lib.toLower c}" else c
          ) chars;
        in
        lib.removePrefix "_" converted;
    in
    lib.concatMapStringsSep "_" to-snake relevant-parts;

  # ────────────────────────────────────────────────────────────────────────────
  # Feature check
  # ────────────────────────────────────────────────────────────────────────────
  require-wasm =
    if wasm-plugin.features.can-load then
      true
    else
      throw (
        builtins.replaceStrings [ "@status@" ] [ wasm-plugin.features.status ] (
          builtins.readFile ./scripts/aleph-wasm-missing-error.txt
        )
      );

  # ────────────────────────────────────────────────────────────────────────────
  # Known module exports (for aleph.import)
  # ────────────────────────────────────────────────────────────────────────────
  known-modules = {
    "Aleph.Packages.Nvidia" = [
      "nccl"
      "cudnn"
      "tensorrt"
      "cutensor"
      "cusparselt"
      "cutlass"
    ];
    "Aleph.Nix.Packages.Nvidia" = [
      "nccl"
      "cudnn"
      "tensorrt"
      "cutensor"
      "cusparselt"
      "cutlass"
    ];
    "Aleph.Packages" = [
      "zlib-ng"
      "fmt"
      "mdspan"
      "cutlass"
      "rapidjson"
      "nlohmann-json"
      "spdlog"
      "catch2"
      "abseil-cpp"
    ];
  };

in
{
  # ════════════════════════════════════════════════════════════════════════════
  # aleph.eval : String -> AttrSet -> NixValue
  # ════════════════════════════════════════════════════════════════════════════
  # Evaluate a typed expression by module path.
  #
  # Examples:
  #   aleph.eval "Aleph.Packages.Nvidia.nccl" {}
  #   aleph.eval "Aleph.Build.withFlags" { pkg = myPkg; flags = ["-O3"]; }
  #
  eval =
    module-path: args:
    assert require-wasm;
    let
      export-name = module-to-export module-path;
      spec = builtins.wasm wasmFile export-name args;
    in
    wasm-plugin.buildFromSpec { inherit spec pkgs stdenv-fn; };

  # ════════════════════════════════════════════════════════════════════════════
  # aleph.import : String -> AttrSet
  # ════════════════════════════════════════════════════════════════════════════
  # Import a module as an attrset of its exports.
  #
  # Examples:
  #   nvidia = aleph.import "Aleph.Packages.Nvidia"
  #   nvidia.nccl  # → derivation
  #
  import =
    module-name:
    assert require-wasm;
    let
      exports =
        known-modules.${module-name}
          or (throw "Unknown module: ${module-name}. Known: ${toString (builtins.attrNames known-modules)}");
      mk-export =
        name:
        let
          export-name = module-to-export "${module-name}.${name}";
          spec = builtins.wasm wasmFile export-name { };
        in
        {
          inherit name;
          value = wasm-plugin.buildFromSpec { inherit spec pkgs stdenv-fn; };
        };
    in
    builtins.listToAttrs (map mk-export exports);

  # ════════════════════════════════════════════════════════════════════════════
  # aleph.spec : String -> AttrSet -> AttrSet
  # ════════════════════════════════════════════════════════════════════════════
  # Get raw spec without building (for debugging/introspection).
  #
  spec =
    module-path: args:
    assert require-wasm;
    let
      export-name = module-to-export module-path;
    in
    builtins.wasm wasmFile export-name args;

  # ════════════════════════════════════════════════════════════════════════════
  # Introspection
  # ════════════════════════════════════════════════════════════════════════════

  inherit (wasm-plugin) features;
  inherit known-modules;
  inherit module-to-export;
}
