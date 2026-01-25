# nix/modules/flake/nixpkgs.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // nixpkgs instantiation //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The sky above the port was the color of television, tuned to a dead
#      channel."
#
#                                                         — Neuromancer
#
# Single nixpkgs instantiation with proper config.
#
# ────────────────────────────────────────────────────────────────────────────────
# WHY `import inputs.nixpkgs` AND NOT `inputs.nixpkgs.legacyPackages.*.extend`?
# ────────────────────────────────────────────────────────────────────────────────
#
# You might think: "use `inputs.nixpkgs.legacyPackages.${system}.appendOverlays`
# to avoid re-instantiating nixpkgs!" This doesn't work because:
#
#   1. nixpkgs config (allowUnfree, cudaSupport, cudaCapabilities) must be set
#      at instantiation time. These are not overlay-able - they're baked into
#      the package set during import.
#
#   2. `legacyPackages` is pre-instantiated with default config (no CUDA, etc.)
#      You cannot retrofit cudaSupport=true onto an existing package set.
#
#   3. For CUDA specifically, the entire cudaPackages hierarchy depends on
#      config.cudaCapabilities being set at import time. See:
#      https://github.com/ConnorBaker/cuda-packages/blob/main/flake.nix
#
# To avoid the "1000 instances" problem (https://zimbatm.com/notes/1000-instances-of-nixpkgs):
# we instantiate ONCE here and all other modules receive pkgs via _module.args.
# The key is: ONE import per (system, config) pair, not one per module.
#
# ────────────────────────────────────────────────────────────────────────────────
#
# We say "nvidia" / "nv", not "cuda". See: docs/languages/nix/philosophy/nvidia-not-cuda.md
# The nixpkgs config keys (cudaSupport, etc.) are upstream API - we can't change those.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph.nixpkgs;
in
{
  _class = "flake";

  # ────────────────────────────────────────────────────────────────────────────
  # // options //
  # ────────────────────────────────────────────────────────────────────────────

  options.aleph.nixpkgs = {
    # NVIDIA GPU support - we say "nv" not "cuda"
    nv = {
      enable = lib.mkEnableOption "NVIDIA GPU support";

      capabilities = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [
          "8.9"
          "9.0"
        ];
        description = "GPU compute capabilities (8.9=Ada, 9.0=Hopper, 12.0=Blackwell)";
      };

      forward-compat = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Enable forward compatibility";
      };
    };

    allow-unfree = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Allow unfree packages";
    };

    overlays = lib.mkOption {
      type = lib.types.listOf lib.types.unspecified;
      default = [ ];
      description = "Additional overlays to apply";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // config //
  # ────────────────────────────────────────────────────────────────────────────

  config.perSystem =
    { system, ... }:
    let
      # Overlay to inject WASM-enabled nix from straylight nix
      alephNixOverlay = _final: _prev: {
        inherit (inputs.nix.packages.${system}) nix;
      };

      # Single nixpkgs instantiation with full config.
      pkgs = import inputs.nixpkgs {
        inherit system;
        # straylight nix provides WASM-enabled nix (builtins.wasm + wasm32-wasip1)
        overlays = [ alephNixOverlay ] ++ cfg.overlays;
        # nixpkgs API uses "cuda" - that's their vocabulary, not ours
        config = {
          allowUnfree = cfg.allow-unfree;
          cudaSupport = cfg.nv.enable;
          cudaCapabilities = cfg.nv.capabilities;
          cudaForwardCompat = cfg.nv.forward-compat;
        };
      };
    in
    {
      # All modules receive this via _module.args.pkgs.
      _module.args.pkgs = lib.mkDefault pkgs;

      # Re-export as legacyPackages so consumers can access our configured nixpkgs:
      #   inputs.aleph.legacyPackages.${system}
      # This is the standard flake output for "the package set".
      legacyPackages = lib.mkDefault pkgs;
    };
}
