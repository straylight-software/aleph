# nix/modules/flake/docs.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                 // docs //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     He knew the edge; he had touched it. It was a gradual thing,
#     building to a long, slow rush. The ice was weakening, the
#     walls were falling, and somewhere, in the darkest heart of
#     the system, he could feel something moving.
#
#                                                         — Neuromancer
#
# Documentation generation. mdBook with ono-sendai or maas themes.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{ config, lib, ... }:
let
  cfg = config.aleph-naught.docs;
in
{
  _class = "flake";

  options.aleph-naught.docs = {
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
      default = ../../../docs;
      description = "mdBook documentation source directory";
    };

    options-src = lib.mkOption {
      type = lib.types.path;
      default = ../../../docs-options;
      description = "NDG options documentation source directory";
    };

    modules = lib.mkOption {
      type = lib.types.listOf lib.types.raw;
      default = [ ];
      description = "NixOS modules to extract options from via NDG";
    };
  };

  config = lib.mkIf cfg.enable {
    perSystem =
      {
        config,
        pkgs,
        system,
        ...
      }:
      let
        ndg-pkg = inputs.ndg.packages.${system}.ndg or null;
        has-ndg = ndg-pkg != null;

        docs-options-drv =
          if cfg.modules == [ ] || !has-ndg then
            pkgs.emptyDirectory
          else
            let
              eval = lib.evalModules {
                inherit (cfg) modules;
              };

              optionsDoc = pkgs.nixosOptionsDoc {
                inherit (eval) options;
                warningsAreErrors = false;
              };

              ndgConfig = pkgs.writeText "ndg.toml" ''
                title = "${cfg.title} Options"
                module_options = "${optionsDoc.optionsJSON}/share/doc/nixos/options.json"
                output_dir = "out"
                stylesheet_path = "${cfg.options-src}/theme/${cfg.theme}.css"

                [search]
                enable = true
              '';
            in
            pkgs.runCommand "aleph-naught-docs-options"
              {
                nativeBuildInputs = [ ndg-pkg ];
              }
              ''
                ${ndg-pkg}/bin/ndg --config-file ${ndgConfig} html -o $out
              '';

        prelude-functions-src = ../../../nix/prelude/functions.nix;
        docs-prelude-drv =
          pkgs.runCommand "aleph-naught-docs-prelude"
            {
              nativeBuildInputs = [ pkgs.nixdoc ];
            }
            ''
              mkdir -p $out
              nixdoc \
                --category prelude \
                --description "Weyl Prelude API Reference" \
                --file ${prelude-functions-src} \
                --prefix "" \
                > $out/prelude-api.md
            '';
      in
      {
        packages = {
          docs-options = docs-options-drv;
          docs-prelude = docs-prelude-drv;

          docs-prose =
            pkgs.runCommand "aleph-naught-docs-prose"
              {
                nativeBuildInputs = [ pkgs.mdbook ];
                inherit (cfg) src;
              }
              ''
                cp -r $src build-src
                chmod -R u+w build-src

                mkdir -p build-src/languages/nix/reference
                cp ${docs-prelude-drv}/prelude-api.md build-src/languages/nix/reference/prelude-api.md

                cd build-src
                mdbook build -d $out
              '';

          docs = pkgs.runCommand "aleph-naught-docs" { } ''
            mkdir -p $out

            cp -r ${config.packages.docs-prose}/. $out/

            chmod -R u+w $out/languages/nix/reference

            mkdir -p $out/languages/nix/reference/options
            if [ -d "${config.packages.docs-options}" ] && [ "$(ls -A ${config.packages.docs-options})" ]; then
              cp -r ${config.packages.docs-options}/. $out/languages/nix/reference/options/
            fi
          '';
        };

        devShells.docs = pkgs.mkShell {
          packages = [
            pkgs.mdbook
            pkgs.nixdoc
          ];

          shellHook = ''
            echo "━━━ aleph-naught docs ━━━"
            echo "mdbook serve     - preview prose docs"
            echo "nix build .#docs - build combined output"
          '';
        };
      };
  };
}
