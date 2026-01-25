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
# Shared options schema for aleph modules.
# Used by both the flake-parts modules and NDG documentation generation.
#
# This is the single source of truth for all aleph option definitions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
let
  # lisp-case aliases for lib functions
  inherit (lib) types;
  mk-enable-option = lib.mkEnableOption;
  mk-option = lib.mkOption;
  list-of = types.listOf;
  null-or = types.nullOr;
  function-to = types.functionTo;
in
{

  # ────────────────────────────────────────────────────────────────────────────
  # // formatter //
  # ────────────────────────────────────────────────────────────────────────────

  formatter = {
    enable = mk-enable-option "aleph formatter" // {
      default = true;
    };
    indent-width = mk-option {
      type = types.int;
      default = 2;
      description = "Indent width in spaces";
    };
    line-length = mk-option {
      type = types.int;
      default = 100;
      description = "Maximum line length";
    };
    enable-check = mk-option {
      type = types.bool;
      default = true;
      description = "Enable flake check for treefmt";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // documentation //
  # ────────────────────────────────────────────────────────────────────────────

  docs = {
    enable = mk-enable-option "documentation generation";
    title = mk-option {
      type = types.str;
      default = "Documentation";
      description = "Documentation title";
    };
    description = mk-option {
      type = types.str;
      default = "";
      description = "Project description";
    };
    theme = mk-option {
      type = types.enum [
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
    src = mk-option {
      type = types.path;
      description = "mdBook documentation source directory";
    };
    options-src = mk-option {
      type = types.path;
      description = "NDG options documentation source directory";
    };
    modules = mk-option {
      type = list-of types.raw;
      default = [ ];
      description = "NixOS modules to extract options from via NDG";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // nixpkgs //
  # ────────────────────────────────────────────────────────────────────────────

  nixpkgs = {
    nv = {
      enable = mk-enable-option "NVIDIA GPU support";
      capabilities = mk-option {
        type = list-of types.str;
        default = [
          "8.9"
          "9.0"
        ];
        description = "NVIDIA capabilities (8.9=Ada, 9.0=Hopper, 12.0=Blackwell)";
      };
      forward-compat = mk-option {
        type = types.bool;
        default = false;
        description = "Enable forward compatibility";
      };
    };
    allow-unfree = mk-option {
      type = types.bool;
      default = true;
      description = "Allow unfree packages";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // overlays //
  # ────────────────────────────────────────────────────────────────────────────

  overlays = {
    enable = mk-enable-option "aleph overlays" // {
      default = true;
    };
    extra = mk-option {
      type = list-of types.raw;
      default = [ ];
      description = "Additional overlays";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // devshell //
  # ────────────────────────────────────────────────────────────────────────────

  devshell = {
    enable = mk-enable-option "aleph devshell";
    nv.enable = mk-enable-option "NVIDIA development tools";
    extra-packages = mk-option {
      type = function-to (list-of types.package);
      default = _: [ ];
      description = "Extra packages (receives pkgs)";
    };
    extra-shell-hook = mk-option {
      type = types.lines;
      default = "";
      description = "Extra shell hook commands";
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // build //
  # ────────────────────────────────────────────────────────────────────────────

  build = {
    enable = mk-enable-option "Buck2 build system integration";
    prelude = {
      enable = mk-option {
        type = types.bool;
        default = true;
        description = "Include straylight-buck2-prelude in flake outputs";
      };
      path = mk-option {
        type = null-or types.path;
        default = null;
        description = ''
          Path to buck2 prelude. If null, uses inputs.buck2-prelude.
          Set this to vendor a local copy.
        '';
      };
    };
    toolchain = {
      cxx = {
        enable = mk-enable-option "C++ toolchain (LLVM 22)";
        c-flags = mk-option {
          type = list-of types.str;
          default = [
            "-O2"
            "-g3"
            "-std=c23"
            "-Wall"
            "-Wextra"
          ];
          description = "C compiler flags";
        };
        cxx-flags = mk-option {
          type = list-of types.str;
          default = [
            "-O2"
            "-g3"
            "-std=c++23"
            "-Wall"
            "-Wextra"
            "-fno-exceptions"
          ];
          description = "C++ compiler flags";
        };
        link-style = mk-option {
          type = types.enum [
            "static"
            "shared"
          ];
          default = "static";
          description = "Default link style";
        };
      };
      nv = {
        enable = mk-enable-option "NVIDIA toolchain (clang + nvidia-sdk)";
        archs = mk-option {
          type = list-of types.str;
          default = [
            "sm_90"
            "sm_100"
            "sm_120"
          ];
          description = ''
            Target NVIDIA architectures:
              sm_90  = Hopper (H100)
              sm_100 = Blackwell (B100, B200)
              sm_120 = Blackwell (B200 full features, requires LLVM 22)
          '';
        };
      };
      haskell = {
        enable = mk-enable-option "Haskell toolchain (GHC from Nix)";
        packages = mk-option {
          type = function-to (list-of types.package);
          default = hp: [
            hp.text
            hp.bytestring
            hp.containers
            hp.aeson
          ];
          description = "Haskell packages for Buck2 toolchain";
        };
      };
      lean.enable = mk-enable-option "Lean 4 toolchain";
      python = {
        enable = mk-enable-option "Python toolchain (with nanobind/pybind11)";
        packages = mk-option {
          type = function-to (list-of types.package);
          default = ps: [
            ps.nanobind
            ps.pybind11
            ps.numpy
          ];
          description = "Python packages for Buck2 toolchain";
        };
      };
    };
    generate-buckconfig = mk-option {
      type = types.bool;
      default = true;
      description = "Generate .buckconfig.local in devshell";
    };
    generate-wrappers = mk-option {
      type = types.bool;
      default = true;
      description = "Generate bin/ wrappers for toolchains";
    };
  };
}
