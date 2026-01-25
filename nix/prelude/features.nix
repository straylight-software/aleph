# nix/prelude/features.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // features //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Wintermute,' the construct said. 'Is this Wintermute? What's
#      going on? Turing? Are you a Turing heat?'
#
#                                                         — Neuromancer
#
# Feature detection and capability requirements. Use to guard experimental
# features and provide actionable error messages when requirements aren't met.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   features.available       attribute set of detected capabilities
#   features.require         assert a feature is available or throw
#   features.when            conditionally evaluate based on feature
#   features.unless          conditionally evaluate when feature absent
#   features.guard           wrap a value, returning null if unavailable
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, pkgs }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Feature Detection
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Attribute set of available features.

    Each key is a feature name, value is true if available.

    # Examples

    ```nix
    available.wasm
    => false  # (unless using straylight-nix)

    available.flakes
    => true   # (on any modern Nix)
    ```
  */
  available = {
    # Nix builtins extensions
    wasm = builtins ? wasm;
    fetchTree = builtins ? fetchTree;
    flakes = builtins ? getFlake;
    fetchClosure = builtins ? fetchClosure;
    trace-verbose = builtins ? traceVerbose;

    # Platform features
    linux = pkgs.stdenv.isLinux;
    darwin = pkgs.stdenv.isDarwin;
    x86_64 = pkgs.stdenv.hostPlatform.isx86_64;
    aarch64 = pkgs.stdenv.hostPlatform.isAarch64;

    # Package availability (checked lazily)
    firecracker = pkgs.stdenv.isLinux && (pkgs ? firecracker);
    cloud-hypervisor = pkgs.stdenv.isLinux && (pkgs ? cloud-hypervisor);
    bubblewrap = pkgs.stdenv.isLinux && (pkgs ? bubblewrap);
    nvidia-driver = pkgs.stdenv.isLinux && (pkgs.config.cudaSupport or false);
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Feature Info (for error messages)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Human-readable information about features.
  */
  info = {
    wasm = {
      name = "WASM plugins";
      description = "Load WebAssembly modules as Nix expressions";
      requires = "straylight-nix with builtins.wasm support";
      docs = "https://github.com/determinate-systems/straylight-nix";
    };

    fetchTree = {
      name = "fetchTree builtin";
      description = "Fetch sources with content-addressability";
      requires = "Nix 2.4+";
      docs = "https://nixos.org/manual/nix/stable/language/builtins.html#builtins-fetchTree";
    };

    flakes = {
      name = "Nix Flakes";
      description = "Reproducible, hermetic Nix expressions";
      requires = "Nix 2.4+ with experimental-features = nix-command flakes";
      docs = "https://nixos.wiki/wiki/Flakes";
    };

    firecracker = {
      name = "Firecracker microVM";
      description = "Lightweight virtualization for secure containers";
      requires = "Linux with KVM support";
      docs = "https://firecracker-microvm.github.io/";
    };

    cloud-hypervisor = {
      name = "Cloud Hypervisor";
      description = "Modern VMM for cloud workloads";
      requires = "Linux with KVM support";
      docs = "https://www.cloudhypervisor.org/";
    };

    bubblewrap = {
      name = "Bubblewrap";
      description = "Unprivileged namespace sandboxing";
      requires = "Linux with user namespaces";
      docs = "https://github.com/containers/bubblewrap";
    };

    nvidia-driver = {
      name = "NVIDIA GPU support";
      description = "CUDA and GPU compute capabilities";
      requires = "Linux with NVIDIA proprietary driver";
      docs = "https://wiki.nixos.org/wiki/Nvidia";
    };
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Requirement Assertions
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Assert that a feature is available, or throw with helpful message.

    # Type

    ```
    require :: String -> Bool
    ```

    # Arguments

    - feature: name of feature (key in `available`)

    # Examples

    ```nix
    require "wasm"
    => error: Feature 'WASM plugins' is not available.
              Requires: straylight-nix with builtins.wasm support
              See: https://github.com/determinate-systems/straylight-nix

    require "flakes"
    => true  # (on modern Nix)
    ```
  */
  require =
    feature:
    if available.${feature} or false then
      true
    else
      let
        feature-info =
          info.${feature} or {
            name = feature;
            description = "Unknown feature";
            requires = "Unknown";
            docs = "";
          };
      in
      throw ''
        Feature '${feature-info.name}' is not available.

        Description: ${feature-info.description}
        Requires: ${feature-info.requires}
        ${lib.optionalString (feature-info.docs != "") "See: ${feature-info.docs}"}
      '';

  /**
    Require multiple features at once.

    # Type

    ```
    require-all :: [String] -> Bool
    ```
  */
  require-all = features: builtins.all require features;

  /**
    Require at least one of the listed features.

    # Type

    ```
    require-any :: [String] -> Bool
    ```
  */
  require-any =
    features:
    if builtins.any (f: available.${f} or false) features then
      true
    else
      throw ''
        None of the required features are available: ${builtins.concatStringsSep ", " features}

        At least one of these features is required.
      '';

  # ─────────────────────────────────────────────────────────────────────────
  # Conditional Evaluation
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Evaluate a thunk only if a feature is available.

    # Type

    ```
    when :: String -> a -> (a | null)
    ```

    # Examples

    ```nix
    when "wasm" { loadWasmPlugin = ...; }
    => null  # (if wasm not available)

    when "flakes" { getFlake = builtins.getFlake; }
    => { getFlake = <primop>; }  # (if flakes available)
    ```
  */
  when = feature: value: if available.${feature} or false then value else null;

  /**
    Evaluate a thunk only if a feature is NOT available.

    Useful for fallbacks and polyfills.

    # Type

    ```
    unless :: String -> a -> (a | null)
    ```
  */
  unless = feature: value: if available.${feature} or false then null else value;

  /**
    Return first available feature's value, or default.

    # Type

    ```
    first-available :: [{ feature :: String; value :: a }] -> a -> a
    ```

    # Examples

    ```nix
    first-available [
      { feature = "wasm"; value = wasmLoader; }
      { feature = "fetchTree"; value = fetchTreeLoader; }
    ] legacyLoader
    ```
  */
  first-available =
    options: default:
    let
      found = lib.findFirst (opt: available.${opt.feature} or false) null options;
    in
    if found == null then default else found.value;

  # ─────────────────────────────────────────────────────────────────────────
  # Guarded Values
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Wrap a value to return null if feature unavailable.

    Unlike `when`, this takes a thunk to avoid evaluation errors.

    # Type

    ```
    guard :: String -> (() -> a) -> (a | null)
    ```

    # Examples

    ```nix
    guard "wasm" (throw "WASM not available")
    => null  # (does not throw)

    guard "flakes" builtins.getFlake
    => <primop>  # (returns the value)
    ```
  */
  guard = feature: thunk: if available.${feature} or false then thunk else null;

  /**
    Create an attribute set that only includes available features.

    # Type

    ```
    filter-available :: AttrSet -> AttrSet
    ```

    # Examples

    ```nix
    filter-available {
      wasm = wasmStuff;
      firecracker = firecrackerStuff;
      flakes = flakeStuff;
    }
    => { flakes = flakeStuff; }  # (only available features)
    ```
  */
  filter-available = lib.filterAttrs (name: _: available.${name} or false);

  # ─────────────────────────────────────────────────────────────────────────
  # Feature Queries
  # ─────────────────────────────────────────────────────────────────────────

  /**
    List all available features.

    # Type

    ```
    list-available :: [String]
    ```
  */
  list-available = lib.attrNames (lib.filterAttrs (_: v: v) available);

  /**
    List all unavailable features.

    # Type

    ```
    list-unavailable :: [String]
    ```
  */
  list-unavailable = lib.attrNames (lib.filterAttrs (_: v: !v) available);

  /**
    Get a diagnostic report of feature availability.

    Useful for debugging and CI logs.

    # Type

    ```
    diagnostic :: String
    ```
  */
  diagnostic =
    let
      format-feature =
        name: avail:
        let
          status = if avail then "[+]" else "[-]";
          feature-info = info.${name} or { inherit name; };
        in
        "${status} ${feature-info.name}";

      lines = lib.mapAttrsToList format-feature available;
    in
    lib.concatStringsSep "\n" ([ "Feature Availability:" ] ++ lines);
}
