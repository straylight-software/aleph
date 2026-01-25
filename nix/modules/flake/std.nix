# nix/modules/flake/std.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                                               // aleph core
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "A year here and he still dreamed of cyberspace, hope fading nightly."
#
#                                                         — Neuromancer
#
# Core aleph module with overlays. This is the entry point that brings
# together nixpkgs instantiation, nix configuration, and the aleph overlay.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph;
  nixpkgs-module = import ./nixpkgs.nix { inherit inputs; };
  nix-conf-module = import ./nix-conf.nix { };
in
{
  _class = "flake";

  imports = [
    nixpkgs-module
    nix-conf-module
  ];

  # ────────────────────────────────────────────────────────────────────────────
  # // options //
  # ────────────────────────────────────────────────────────────────────────────

  options.aleph.overlays = {
    enable = lib.mkEnableOption "aleph overlays" // {
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
      straylight-nix-overlay = _final: _prev: {
        inherit (inputs.nix.packages.${system}) nix;
      };

      pkgs-with-overlays = import inputs.nixpkgs {
        inherit system;

        # nixpkgs API uses "cuda" - that's their vocabulary, not ours
        # Use string access for external camelCase API names
        config = {
          "allowUnfree" = cfg.nixpkgs.allow-unfree;
          "cudaSupport" = cfg.nixpkgs.nv.enable;
          "cudaCapabilities" = cfg.nixpkgs.nv.capabilities;
          "cudaForwardCompat" = cfg.nixpkgs.nv.forward-compat;
        };

        # straylight nix provides WASM-enabled nix (builtins.wasm + wasm32-wasip1)
        overlays = [
          straylight-nix-overlay
          (import ../../overlays inputs).flake.overlays.default
        ]
        ++ cfg.overlays.extra;
      };
    in
    lib.mkIf cfg.overlays.enable {
      _module.args.pkgs = lib.mkForce pkgs-with-overlays;

      # Re-export configured pkgs with overlays as legacyPackages
      # This lets consumers access aleph.prelude, aleph.stdenv, etc.:
      #   inputs.aleph.legacyPackages.${system}.aleph.prelude
      legacyPackages = lib.mkForce pkgs-with-overlays;
    };
}
