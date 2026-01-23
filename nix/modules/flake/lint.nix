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
# Typed Haskell scripts replace bash. The scripts take a configs directory
# as an argument, which is baked in via a wrapper script at build time.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_:
let
  # Individual config file paths (for flake.lintConfigs export)
  lintConfigs = {
    clang-format = ../../configs/.clang-format;
    clang-tidy = ../../configs/.clang-tidy;
    ruff = ../../configs/ruff.toml;
    biome = ../../configs/biome.json;
    stylua = ../../configs/.stylua.toml;
    rustfmt = ../../configs/.rustfmt.toml;
    taplo = ../../configs/taplo.toml;
  };

  # Script source directory
  scriptSrc = ../../../src/tools/scripts;
in
{ config, lib, ... }:
let
  cfg = config.aleph-naught.lint;
in
{
  _class = "flake";

  options.aleph-naught.lint = {
    enable = lib.mkEnableOption "aleph-naught lint configs" // {
      default = true;
    };
  };

  config = lib.mkIf cfg.enable {
    flake.lintConfigs = lintConfigs;

    perSystem =
      { pkgs, ... }:
      let
        # Create a configs directory derivation with all lint configs
        configsDir = pkgs.linkFarm "straylight-lint-configs" [
          {
            name = ".clang-format";
            path = lintConfigs.clang-format;
          }
          {
            name = ".clang-tidy";
            path = lintConfigs.clang-tidy;
          }
          {
            name = "ruff.toml";
            path = lintConfigs.ruff;
          }
          {
            name = "biome.json";
            path = lintConfigs.biome;
          }
          {
            name = ".stylua.toml";
            path = lintConfigs.stylua;
          }
          {
            name = ".rustfmt.toml";
            path = lintConfigs.rustfmt;
          }
          {
            name = "taplo.toml";
            path = lintConfigs.taplo;
          }
        ];

        # Haskell dependencies for Weyl.Script
        hsDeps =
          p: with p; [
            megaparsec
            text
            shelly
            foldl
            aeson
            directory
            cryptonite
            memory
            unordered-containers
            vector
            unix
            async
            bytestring
          ];

        ghcWithDeps = pkgs.haskellPackages.ghcWithPackages hsDeps;

        # Build a lint script
        # The script takes configs-dir as first argument, which we bake in via wrapper
        mkLintScript =
          name:
          pkgs.stdenv.mkDerivation {
            inherit name;
            src = scriptSrc;
            dontUnpack = true;

            nativeBuildInputs = [
              ghcWithDeps
              pkgs.makeWrapper
            ];

            buildPhase = ''
              runHook preBuild
              # Use -hidir and -odir to avoid writing to read-only Nix store
              ghc -O2 -Wall -Wno-unused-imports \
                -hidir . -odir . \
                -i$src -o ${name} $src/${name}.hs
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/bin
              cp ${name} $out/bin/${name}-unwrapped

              # Wrap with configs directory baked in
              makeWrapper $out/bin/${name}-unwrapped $out/bin/${name} \
                --add-flags "${configsDir}"
              runHook postInstall
            '';

            meta = {
              description = "Lint configuration script for code formatting and style checks";
            };
          };
      in
      {
        packages.lint-init = mkLintScript "lint-init";
        packages.lint-link = mkLintScript "lint-link";
      };
  };
}
