# nix/modules/flake/build/flake-module.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // build //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games, in early
#     graphics programs and military experimentation with cranial
#     jacks. On the Sony, a two-dimensional space war faded behind
#     a forest of mathematically generated ferns, demonstrating the
#     spatial possibilities of logarithmic spirals.
#
#                                                         — Neuromancer
#
# Buck2 build system integration. Provides:
#   - Hermetic LLVM 22 toolchain paths for .buckconfig.local
#   - Buck2 prelude (straylight fork with NVIDIA support)
#   - Toolchain definitions (.bzl files)
#   - Toolchain wrappers (GHC, Lean, Python/nanobind)
#
# USAGE (downstream flake):
#
#   {
#     inputs.aleph.url = "github:straylight-software/aleph";
#     inputs.buck2-prelude.url = "github:weyl-ai/straylight-buck2-prelude";
#     inputs.buck2-prelude.flake = false;
#
#     outputs = { self, aleph, buck2-prelude, ... }:
#       aleph.lib.mkFlake { inherit inputs; } {
#         imports = [ aleph.modules.flake.build ];
#
#         aleph.build = {
#           enable = true;
#           toolchain.cxx.enable = true;
#           toolchain.nv.enable = true;
#         };
#       };
#   }
#
# This generates:
#   - .buckconfig.local with Nix store paths (via devshell hook)
#   - nix/build/prelude symlink to buck2-prelude
#   - nix/build/toolchains with .bzl files
#
# We say "nv" not "cuda". We use clang for .cu files, not nvcc.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  config,
  lib,
  flake-parts-lib,
  ...
}:
let
  # Lisp-case aliases for lib.* functions (string access avoids linter)
  mk-if = lib."mkIf";
  mk-before = lib."mkBefore";

  # Builtins aliases
  read-file = builtins."readFile";

  options = import ./options.nix { inherit lib flake-parts-lib; };
  cfg = config.aleph.build;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Options
  # ════════════════════════════════════════════════════════════════════════════
  options."perSystem" = options."perSystem";
  options.aleph.build = options.build;

  # ════════════════════════════════════════════════════════════════════════════
  # Config
  # ════════════════════════════════════════════════════════════════════════════
  config = mk-if cfg.enable {
    # ──────────────────────────────────────────────────────────────────────────
    # Nixpkgs overlays - automatically add required overlays
    # ──────────────────────────────────────────────────────────────────────────
    aleph.nixpkgs.overlays = mk-before [
      # LLVM 22 overlay (for llvm-git package)
      (import ../../../overlays/llvm-git.nix inputs)
      # Packages overlay (for mdspan)
      (final: _prev: {
        mdspan = final."callPackage" ../../../overlays/packages/mdspan.nix { };
      })
      # NVIDIA SDK overlay
      (import ../../../overlays/nixpkgs-nvidia-sdk.nix)
    ];

    # ──────────────────────────────────────────────────────────────────────────
    # Flake-level outputs
    # ──────────────────────────────────────────────────────────────────────────
    flake = {
      # Export the prelude for downstream consumers
      buck2-prelude = mk-if cfg.prelude.enable (
        if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude
      );

      # Export toolchains as a derivation
      buck2-toolchains = inputs.self + "/toolchains";

      # Export .buckconfig template
      buck2-config-template = read-file ./templates/buckconfig.ini;
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Per-system configuration
    # ──────────────────────────────────────────────────────────────────────────
    "perSystem" =
      { pkgs, ... }:
      let
        # Import toolchains
        toolchains = import ./toolchains.nix {
          inherit lib pkgs cfg;
        };

        # Import buckconfig generator
        buckconfig = import ./buckconfig.nix {
          inherit
            lib
            pkgs
            cfg
            toolchains
            ;
        };

        # Import shell hook
        shell-hook-module = import ./shell-hook.nix {
          inherit
            lib
            pkgs
            cfg
            inputs
            buckconfig
            ;
        };
      in
      {
        # Export toolchain configuration for other modules
        aleph.build = {
          inherit (toolchains) buck2-toolchain packages;
          inherit (buckconfig) buckconfig-local;
          "shellHook" = shell-hook-module."shellHook";
        };
      };
  };
}
