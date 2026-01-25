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
  lint-configs = {
    clang-format = ../../configs/.clang-format;
    clang-tidy = ../../configs/.clang-tidy;
    ruff = ../../configs/ruff.toml;
    biome = ../../configs/biome.json;
    stylua = ../../configs/.stylua.toml;
    rustfmt = ../../configs/.rustfmt.toml;
    taplo = ../../configs/taplo.toml;
  };

  # Script source directory
  script-src = ../../../src/tools/scripts;
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
    flake.lintConfigs = lint-configs;

    perSystem =
      { pkgs, ... }:
      let
        # Create a configs directory derivation with all lint configs
        configs-dir = pkgs.linkFarm "straylight-lint-configs" [
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

        # Haskell dependencies for Weyl.Script
        hs-deps =
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

        ghc-with-deps = pkgs.haskellPackages.ghcWithPackages hs-deps;

        # Build a lint script
        # The script takes configs-dir as first argument, which we bake in via wrapper
        mk-lint-script =
          name:
          pkgs.stdenv.mkDerivation {
            inherit name;
            src = script-src;
            dontUnpack = true;

            nativeBuildInputs = [
              ghc-with-deps
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
                --add-flags "${configs-dir}"
              runHook postInstall
            '';

            meta = {
              description = "Lint configuration script for code formatting and style checks";
            };
          };
      in
      {
        packages.lint-init = mk-lint-script "lint-init";
        packages.lint-link = mk-lint-script "lint-link";
      };
  };
}
