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
  stdenvFn ? pkgs.stdenv.mkDerivation,
}:
let
  # Import the WASM plugin infrastructure
  wasmPlugin = import ./wasm-plugin.nix {
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
  moduleToExport =
    modulePath:
    let
      parts = lib.splitString "." modulePath;
      # Take everything after known prefixes
      relevantParts =
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
      toSnake =
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
    lib.concatMapStringsSep "_" toSnake relevantParts;

  # ────────────────────────────────────────────────────────────────────────────
  # Feature check
  # ────────────────────────────────────────────────────────────────────────────
  requireWasm =
    if wasmPlugin.features.can-load then
      true
    else
      throw (
        builtins.replaceStrings [ "@status@" ] [ wasmPlugin.features.status ] (
          builtins.readFile ./scripts/aleph-wasm-missing-error.txt
        )
      );

  # ────────────────────────────────────────────────────────────────────────────
  # Known module exports (for aleph.import)
  # ────────────────────────────────────────────────────────────────────────────
  knownModules = {
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
    modulePath: args:
    assert requireWasm;
    let
      exportName = moduleToExport modulePath;
      spec = builtins.wasm wasmFile exportName args;
    in
    wasmPlugin.buildFromSpec { inherit spec pkgs stdenvFn; };

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
    moduleName:
    assert requireWasm;
    let
      exports =
        knownModules.${moduleName}
          or (throw "Unknown module: ${moduleName}. Known: ${toString (builtins.attrNames knownModules)}");
      mkExport =
        name:
        let
          exportName = moduleToExport "${moduleName}.${name}";
          spec = builtins.wasm wasmFile exportName { };
        in
        {
          inherit name;
          value = wasmPlugin.buildFromSpec { inherit spec pkgs stdenvFn; };
        };
    in
    builtins.listToAttrs (map mkExport exports);

  # ════════════════════════════════════════════════════════════════════════════
  # aleph.spec : String -> AttrSet -> AttrSet
  # ════════════════════════════════════════════════════════════════════════════
  # Get raw spec without building (for debugging/introspection).
  #
  spec =
    modulePath: args:
    assert requireWasm;
    let
      exportName = moduleToExport modulePath;
    in
    builtins.wasm wasmFile exportName args;

  # ════════════════════════════════════════════════════════════════════════════
  # Introspection
  # ════════════════════════════════════════════════════════════════════════════

  inherit (wasmPlugin) features;
  inherit knownModules;
  inherit moduleToExport;
}
