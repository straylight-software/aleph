# nix/modules/flake/formatter.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // formatter //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Case's virus had bored a window through the library's command
#     ice. He punched himself through and found an address.
#
#                                                         — Neuromancer
#
# Treefmt integration. Unified formatting across all languages.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph.formatter;
in
{
  _class = "flake";
  imports = [ inputs.treefmt-nix.flakeModule ];

  options.aleph.formatter = {
    enable = lib.mkEnableOption "aleph formatter" // {
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

  config = lib.mkIf cfg.enable {
    perSystem =
      { system, ... }:
      {
        treefmt = {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
          programs.deadnix = {
            enable = true;
            excludes = [ "nix/templates/*" ];
          };
          programs.statix.enable = true;

          programs.shfmt = {
            enable = true;
            "indent_size" = cfg.indent-width;
          };

          settings.formatter = {
            aleph-lint = {
              command = lib.getExe inputs.self.packages.${system}.aleph-lint;
              includes = [ "*.nix" ];
            };
          };

          programs.ruff-format = {
            enable = true;
            "lineLength" = cfg.line-length;
          };

          programs.ruff-check.enable = true;

          programs.clang-format = {
            enable = true;
            includes = [
              "*.c"
              "*.h"
              "*.cpp"
              "*.hpp"
              "*.cu"
              "*.cuh"
              "*.proto"
            ];
          };

          programs.biome = {
            enable = true;
            settings.formatter = {
              "indentStyle" = "space";
              "indentWidth" = cfg.indent-width;
              "lineWidth" = cfg.line-length;
            };
            settings.css.linter.enabled = false;
          };

          programs.stylua = {
            enable = true;
            settings = {
              "column_width" = cfg.line-length;
              "indent_type" = "Spaces";
              "indent_width" = cfg.indent-width;
            };
          };

          programs.taplo.enable = true;
          programs.yamlfmt.enable = true;
          programs.mdformat.enable = true;
          programs.fourmolu.enable = true;
          # hlint disabled - it's a linter, not a formatter, and treefmt-nix
          # doesn't support the config file needed to suppress suggestions
          # programs.hlint.enable = true;
          programs.just.enable = true;
          programs.keep-sorted.enable = true;
          "flakeCheck" = cfg.enable-check;
        };
      };
  };
}
