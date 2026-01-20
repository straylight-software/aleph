# nix/modules/flake/options-schema.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // options schema //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "He'd found himself in a disused netrunners' bazaar, somewhere off in
#      the sprawl's fringes, where old-style cyberspace operators maintained
#      their markets."
#
#                                                         — Neuromancer
#
# Shared options schema for aleph-naught modules.
# Used by both the flake-parts modules and NDG documentation generation.
#
# This is the single source of truth for all aleph-naught option definitions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
{
  # ────────────────────────────────────────────────────────────────────────────
  # // formatter //
  # ────────────────────────────────────────────────────────────────────────────

  formatter = {
    enable = lib.mkEnableOption "aleph-naught formatter" // {
      default = true;
    };
    indent-width = lib.mkOption {
      type = lib.types.int;
      default = 2;
      description = "Indent width in spaces";
    };
    line-length = lib.mkOption {
      type = lib.types.int;
      default = 100;
      description = "Maximum line length";
    };
    enable-check = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable flake check for treefmt";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // documentation //
  # ────────────────────────────────────────────────────────────────────────────

  docs = {
    enable = lib.mkEnableOption "documentation generation";
    title = lib.mkOption {
      type = lib.types.str;
      default = "Documentation";
      description = "Documentation title";
    };
    description = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Project description";
    };
    theme = lib.mkOption {
      type = lib.types.enum [
        "ono-sendai"
        "maas"
      ];
      default = "ono-sendai";
      description = ''
        Documentation theme.
        - ono-sendai: Dark mode (cyberdeck interface)
        - maas: Light mode (clean room schematics)
      '';
    };
    src = lib.mkOption {
      type = lib.types.path;
      description = "mdBook documentation source directory";
    };
    options-src = lib.mkOption {
      type = lib.types.path;
      description = "NDG options documentation source directory";
    };
    modules = lib.mkOption {
      type = lib.types.listOf lib.types.raw;
      default = [ ];
      description = "NixOS modules to extract options from via NDG";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // nixpkgs //
  # ────────────────────────────────────────────────────────────────────────────

  nixpkgs = {
    nv = {
      enable = lib.mkEnableOption "NVIDIA GPU support";
      capabilities = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [
          "8.9"
          "9.0"
        ];
        description = "NVIDIA capabilities (8.9=Ada, 9.0=Hopper, 12.0=Blackwell)";
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
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // overlays //
  # ────────────────────────────────────────────────────────────────────────────

  overlays = {
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
  # // devshell //
  # ────────────────────────────────────────────────────────────────────────────

  devshell = {
    enable = lib.mkEnableOption "aleph-naught devshell";
    nv.enable = lib.mkEnableOption "NVIDIA development tools";
    extra-packages = lib.mkOption {
      type = lib.types.functionTo (lib.types.listOf lib.types.package);
      default = _: [ ];
      description = "Extra packages (receives pkgs)";
    };
    extra-shell-hook = lib.mkOption {
      type = lib.types.lines;
      default = "";
      description = "Extra shell hook commands";
    };
  };
}
