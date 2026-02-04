# nix/modules/flake/lint.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                 // lint //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The Flatline laughed. 'Hey, you know how old I am? You know
#      how many times I been here, watching punks like you flash
#      their silicon? I got news for you, kid.'
#
#                                                         — Neuromancer
#
# Lint config files. clang-format, clang-tidy, ruff, biome, stylua, etc.
#
# Typed Haskell scripts are built via Buck2 and exposed as packages.
# The scripts take a configs directory as an argument, which is baked
# in via a wrapper script.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_:
let
  # Individual config file paths (for flake.lintConfigs export)
  # :: { biome : Path, clang-format : Path, clang-tidy : Path, ruff : Path, rustfmt : Path, stylua : Path, taplo : Path }
  # :: Path
  # :: Path
  # :: Path
  # :: Path
  # :: Path
  # :: Path
  # :: Path
  lint-configs = {
    clang-format = ../../configs/.clang-format;
    clang-tidy = ../../configs/.clang-tidy;
    ruff = ../../configs/ruff.toml;
    # :: t5
    biome = ../../configs/biome.json;
    stylua = ../../configs/.stylua.toml;
    # :: "flake"
    rustfmt = ../../configs/.rustfmt.toml;
    taplo = ../../configs/taplo.toml;
  };
in
{ config, lib, ... }:
let
  cfg = config.aleph.lint;
# :: t15
in
{
  # :: { pkgs : t9 } | ... -> {}
  _class = "flake";

  options.aleph.lint = {
    # :: t14
    enable = lib.mkEnableOption "aleph lint configs" // {
      # :: ".clang-format"
      # :: Path
      default = true;
    };
  # :: ".clang-tidy"
  # :: Path
  };

  # :: "ruff.toml"
  # :: Path
  config = lib.mkIf cfg.enable {
    flake.lintConfigs = lint-configs;
# :: "biome.json"
# :: Path

    perSystem =
      # :: ".stylua.toml"
      # :: Path
      { pkgs, ... }:
      let
        # :: ".rustfmt.toml"
        # :: Path
        # Create a configs directory derivation with all lint configs
        configs-dir = pkgs.linkFarm "aleph-lint-configs" [
          # :: "taplo.toml"
          # :: Path
          {
            name = ".clang-format";
            path = lint-configs.clang-format;
          }
          {
            name = ".clang-tidy";
            path = lint-configs.clang-tidy;
          }
          {
            name = "ruff.toml";
            path = lint-configs.ruff;
          }
          {
            name = "biome.json";
            path = lint-configs.biome;
          }
          {
            name = ".stylua.toml";
            path = lint-configs.stylua;
          }
          {
            name = ".rustfmt.toml";
            path = lint-configs.rustfmt;
          }
          {
            name = "taplo.toml";
            path = lint-configs.taplo;
          }
        ];

        # Pre-built scripts from overlay (compiled via GHC with Aleph.Script)
        inherit (pkgs.aleph.script) compiled;

        # Wrap a pre-built script with configs directory baked in
        wrap-lint-script =
          name:
          pkgs.runCommand name
            {
              nativeBuildInputs = [ pkgs.makeWrapper ];
              meta.description = "Lint configuration script for code formatting and style checks";
            }
            ''
              mkdir -p $out/bin
              makeWrapper ${compiled.${name}}/bin/${name} $out/bin/${name} \
                --add-flags "${configs-dir}"
            '';
      in
      {
        packages.lint-init = wrap-lint-script "lint-init";
        packages.lint-link = wrap-lint-script "lint-link";
      };
  };
}
