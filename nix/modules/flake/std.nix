# nix/modules/flake/std.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                                               // aleph-naught core
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "A year here and he still dreamed of cyberspace, hope fading nightly."
#
#                                                         — Neuromancer
#
# Core aleph-naught module with overlays. This is the entry point that brings
# together nixpkgs instantiation, nix configuration, and the straylight overlay.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph-naught;
  nixpkgsModule = import ./nixpkgs.nix { inherit inputs; };
  nixconfModule = import ./nix-conf.nix { };
in
{
  _class = "flake";

  imports = [
    nixpkgsModule
    nixconfModule
  ];

  # ────────────────────────────────────────────────────────────────────────────
  # // options //
  # ────────────────────────────────────────────────────────────────────────────

  options.aleph-naught.overlays = {
    enable = lib.mkEnableOption "aleph-naught overlays" // {
      default = true;
    };

    extra = lib.mkOption {
      type = lib.types.listOf lib.types.raw;
      default = [ ];
      description = "Additional overlays";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // config //
  # ────────────────────────────────────────────────────────────────────────────

  config.perSystem =
    { system, ... }:
    let
      # Overlay to inject WASM-enabled nix from straylight nix
      straylightNixOverlay = _final: _prev: {
        inherit (inputs.nix.packages.${system}) nix;
      };

      pkgsWithOverlays = import inputs.nixpkgs {
        inherit system;

        # nixpkgs API uses "cuda" - that's their vocabulary, not ours
        config = {
          allowUnfree = cfg.nixpkgs.allow-unfree;
          cudaSupport = cfg.nixpkgs.nv.enable;
          cudaCapabilities = cfg.nixpkgs.nv.capabilities;
          cudaForwardCompat = cfg.nixpkgs.nv.forward-compat;
        };

        # straylight nix provides WASM-enabled nix (builtins.wasm + wasm32-wasip1)
        overlays = [
          straylightNixOverlay
          (import ../../overlays inputs).flake.overlays.default
        ]
        ++ cfg.overlays.extra;
      };
    in
    lib.mkIf cfg.overlays.enable {
      _module.args.pkgs = lib.mkForce pkgsWithOverlays;

      # Re-export configured pkgs with overlays as legacyPackages
      # This lets consumers access straylight.prelude, straylight.stdenv, etc.:
      #   inputs.aleph-naught.legacyPackages.${system}.straylight.prelude
      legacyPackages = lib.mkForce pkgsWithOverlays;
    };
}
