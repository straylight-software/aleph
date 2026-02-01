# nix/modules/flake/formatter.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                                         // formatter // module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Case's virus had bored a window through the library's command ice.
#      He punched himself through and found an address."
#
#                                                               — Neuromancer
#
# Treefmt integration. Unified formatting and linting across all languages.
#
# Pipeline for C/C++:
#   1. clang-format (mechanical layout, rewrites files)
#   2. clang-tidy (semantic lint, warnings-as-errors)
#   3. cppcheck (deep flow analysis)
#   4. aleph-grep (straylight-specific AST patterns)
#
# See: docs/guidelines/03-cpp.md for rationale.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph.formatter;

  # ──────────────────────────────────────────────────────────────────────────
  #                                                        // cpp // file // patterns
  # ──────────────────────────────────────────────────────────────────────────

  cpp-includes = [
    "*.c"
    "*.h"
    "*.cpp"
    "*.hpp"
    "*.cc"
    "*.hh"
    "*.cxx"
    "*.hxx"
    "*.cu"
    "*.cuh"
  ];

in
{
  _class = "flake";
  imports = [ inputs.treefmt-nix.flakeModule ];

  # ════════════════════════════════════════════════════════════════════════════
  #                                                                   // options
  # ════════════════════════════════════════════════════════════════════════════

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

    # ──────────────────────────────────────────────────────────────────────────
    #                                                        // cpp // options
    # ──────────────────────────────────────────────────────────────────────────

    cpp = {
      enable-clang-tidy = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable clang-tidy semantic linting";
      };

      enable-cppcheck = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable cppcheck flow analysis";
      };

      enable-aleph-grep = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable aleph-grep AST pattern checks";
      };

      compile-commands-path = lib.mkOption {
        type = lib.types.str;
        default = "compile_commands.json";
        description = "Path to compile_commands.json for clang-tidy";
      };
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  #                                                                    // config
  # ════════════════════════════════════════════════════════════════════════════

  config = lib.mkIf cfg.enable {
    perSystem =
      { pkgs, system, ... }:
      let
        # ────────────────────────────────────────────────────────────────────────
        #                                                      // llvm // toolchain
        # ────────────────────────────────────────────────────────────────────────
        #
        # Use our custom LLVM 22 from git for clang-format and clang-tidy.
        # Falls back to nixpkgs llvmPackages_19 if llvm-git not available.
        #
        # ────────────────────────────────────────────────────────────────────────

        llvm-pkg = pkgs.llvm-git or pkgs.llvmPackages_19.clang-tools;

        # ────────────────────────────────────────────────────────────────────────
        #                                                   // clang-tidy // wrapper
        # ────────────────────────────────────────────────────────────────────────

        clang-tidy-wrapper = pkgs.writeShellApplication {
          name = "clang-tidy-check";
          runtimeInputs = [ ];
          text = ''
            export COMPILE_COMMANDS_PATH="${cfg.cpp.compile-commands-path}"
            export CLANG_TIDY_BIN="${llvm-pkg}/bin/clang-tidy"
            ${builtins.readFile ./scripts/clang-tidy-check.sh}
          '';
        };

        # ────────────────────────────────────────────────────────────────────────
        #                                                     // cppcheck // wrapper
        # ────────────────────────────────────────────────────────────────────────

        cppcheck-wrapper = pkgs.writeShellApplication {
          name = "cppcheck-check";
          runtimeInputs = [ ];
          text = ''
            export CPPCHECK_BIN="${pkgs.cppcheck}/bin/cppcheck"
            ${builtins.readFile ./scripts/cppcheck-check.sh}
          '';
        };

        # ────────────────────────────────────────────────────────────────────────
        #                                                   // aleph-grep // wrapper
        # ────────────────────────────────────────────────────────────────────────
        #
        # n.b. ast-grep --rule takes a single file, so we run each rule separately
        # and exit on first failure
        #
        # ────────────────────────────────────────────────────────────────────────

        aleph-grep-cpp-wrapper = pkgs.writeShellApplication {
          name = "aleph-grep-cpp";
          runtimeInputs = [ ];
          text = ''
            export RULES_DIR="${inputs.self}/linter/rules"
            export AST_GREP_BIN="${pkgs.ast-grep}/bin/ast-grep"
            ${builtins.readFile ./scripts/aleph-grep-cpp.sh}
          '';
        };

      in
      {
        treefmt = {
          projectRootFile = "flake.nix";

          # ────────────────────────────────────────────────────────────────────────
          #                                                            // nix // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.nixfmt.enable = true;

          programs.deadnix = {
            enable = true;
            excludes = [ "nix/templates/*" ];
          };

          programs.statix.enable = true;

          # ────────────────────────────────────────────────────────────────────────
          #                                                          // shell // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.shfmt = {
            enable = true;
            "indent_size" = cfg.indent-width;
          };

          # ────────────────────────────────────────────────────────────────────────
          #                                                         // python // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.ruff-format = {
            enable = true;
            "lineLength" = cfg.line-length;
          };

          programs.ruff-check.enable = true;

          # ────────────────────────────────────────────────────────────────────────
          #                                                            // cpp // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.clang-format = {
            enable = true;
            package = llvm-pkg;
            includes = cpp-includes ++ [ "*.proto" ];
          };

          # ────────────────────────────────────────────────────────────────────────
          #                                                             // js // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.biome = {
            enable = true;
            settings.formatter = {
              "indentStyle" = "space";
              "indentWidth" = cfg.indent-width;
              "lineWidth" = cfg.line-length;
            };
            settings.css.linter.enabled = false;
          };

          # ────────────────────────────────────────────────────────────────────────
          #                                                            // lua // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.stylua = {
            enable = true;
            settings = {
              "column_width" = cfg.line-length;
              "indent_type" = "Spaces";
              "indent_width" = cfg.indent-width;
            };
          };

          # ────────────────────────────────────────────────────────────────────────
          #                                                        // haskell // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.fourmolu.enable = true;
          # n.b. hlint disabled — it's a linter, not a formatter, and treefmt-nix
          # doesn't support the config file needed to suppress suggestions

          # ────────────────────────────────────────────────────────────────────────
          #                                                          // other // lint
          # ────────────────────────────────────────────────────────────────────────

          programs.taplo.enable = true;
          programs.yamlfmt.enable = true;
          programs.mdformat.enable = true;
          programs.just.enable = true;
          programs.keep-sorted.enable = true;

          # ────────────────────────────────────────────────────────────────────────
          #                                                 // custom // formatters
          # ────────────────────────────────────────────────────────────────────────

          settings.formatter = {
            # ── nix // aleph-lint ────────────────────────────────────────────────

            aleph-lint = {
              command = lib.getExe inputs.self.packages.${system}.aleph-lint;
              includes = [ "*.nix" ];
            };

            # ── cpp // clang-tidy ────────────────────────────────────────────────

            clang-tidy = lib.mkIf cfg.cpp.enable-clang-tidy {
              command = clang-tidy-wrapper;
              includes = cpp-includes;
            };

            # ── cpp // cppcheck ──────────────────────────────────────────────────

            cppcheck = lib.mkIf cfg.cpp.enable-cppcheck {
              command = cppcheck-wrapper;
              includes = cpp-includes;
            };

            # ── cpp // aleph-grep ────────────────────────────────────────────────

            aleph-grep-cpp = lib.mkIf cfg.cpp.enable-aleph-grep {
              command = aleph-grep-cpp-wrapper;
              includes = cpp-includes;
            };
          };

          "flakeCheck" = cfg.enable-check;
        };
      };
  };
}
