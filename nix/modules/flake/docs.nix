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
  # ── lisp-case aliases ──────────────────────────────────────────────────────
  mk-enable-option = lib.mkEnableOption;
  mk-option = lib.mkOption;
  mk-if = lib.mkIf;
  list-of = lib.types.listOf;
  eval-modules = lib.evalModules;

  cfg = config.aleph.docs;
in
{
  _class = "flake";

  options.aleph.docs = {
    enable = mk-enable-option "documentation generation";

    title = mk-option {
      type = lib.types.str;
      default = "Documentation";
      description = "Documentation title";
    };

    description = mk-option {
      type = lib.types.str;
      default = "";
      description = "Project description";
    };

    theme = mk-option {
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

    src = mk-option {
      type = lib.types.path;
      default = ../../../docs;
      description = "mdBook documentation source directory";
    };

    options-src = mk-option {
      type = lib.types.path;
      default = ../../../docs-options;
      description = "NDG options documentation source directory";
    };

    modules = mk-option {
      type = list-of lib.types.raw;
      default = [ ];
      description = "NixOS modules to extract options from via NDG";
    };
  };

  config = mk-if cfg.enable {
    perSystem =
      {
        config,
        pkgs,
        system,
        ...
      }:
      let
        # ── lisp-case aliases ────────────────────────────────────────────────
        empty-directory = pkgs.emptyDirectory;
        nixos-options-doc = pkgs.nixosOptionsDoc;
        write-text = pkgs.writeText;
        run-command = pkgs.runCommand;
        mk-shell = pkgs.mkShell;

        has-ndg = inputs ? ndg && inputs.ndg.packages.${system} ? ndg;
        ndg-pkg = if has-ndg then inputs.ndg.packages.${system}.ndg else null;

        docs-options-drv =
          if cfg.modules == [ ] || !has-ndg then
            empty-directory
          else
            let
              eval = eval-modules {
                inherit (cfg) modules;
              };

              options-doc = nixos-options-doc {
                inherit (eval) options;
                warningsAreErrors = false;
              };

              ndg-config = write-text "ndg.toml" ''
                title = "${cfg.title} Options"
                module_options = "${options-doc.optionsJSON}/share/doc/nixos/options.json"
                output_dir = "out"
                stylesheet_path = "${cfg.options-src}/theme/${cfg.theme}.css"

                [search]
                enable = true
              '';
            in
            run-command "aleph-docs-options"
              {
                nativeBuildInputs = [ ndg-pkg ];
              }
              ''
                ${ndg-pkg}/bin/ndg --config-file ${ndg-config} html -o $out
              '';

        prelude-functions-src = ../../../nix/prelude/functions.nix;
        docs-prelude-drv =
          run-command "aleph-docs-prelude"
            {
              nativeBuildInputs = [ pkgs.nixdoc ];
            }
            ''
              mkdir -p $out
              nixdoc \
                --category prelude \
                --description "Aleph Prelude API Reference" \
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
            run-command "aleph-docs-prose"
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

          docs = run-command "aleph-docs" { } ''
            mkdir -p $out

            cp -r ${config.packages.docs-prose}/. $out/

            chmod -R u+w $out/languages/nix/reference

            mkdir -p $out/languages/nix/reference/options
            if [ -d "${config.packages.docs-options}" ] && [ "$(ls -A ${config.packages.docs-options})" ]; then
              cp -r ${config.packages.docs-options}/. $out/languages/nix/reference/options/
            fi
          '';
        };

        devShells.docs = mk-shell {
          packages = [
            pkgs.mdbook
            pkgs.nixdoc
          ];

          shellHook = ''
            echo "━━━ aleph docs ━━━"
            echo "mdbook serve     - preview prose docs"
            echo "nix build .#docs - build combined output"
          '';
        };
      };
  };
}
