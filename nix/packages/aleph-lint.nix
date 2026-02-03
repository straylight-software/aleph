{
  pkgs,
  tree-sitter-grammars,
  tree-sitter,
  ast-grep,
  lib,
  dhall,
}:
let
  inherit (pkgs.aleph) write-shell-application run-command;
  linter-src = ../../linter;

  ast-grep-config =
    run-command "ast-grep-config"
      {
        native-build-inputs = [ dhall ];
      }
      ''
        mkdir -p $out
        export XDG_CACHE_HOME=$TMPDIR/.cache
        mkdir -p $XDG_CACHE_HOME/dhall

        ln -sf ${pkgs.dhallPackages.Prelude}/.cache/dhall/* $XDG_CACHE_HOME/dhall/

        dhall to-directory-tree --file ${linter-src}/generate.dhall --output $out
      '';
in
write-shell-application {
  name = "aleph-lint";
  runtime-inputs = [
    ast-grep
    tree-sitter
    tree-sitter-grammars.tree-sitter-nix
  ];

  derivation-args.postCheck = ''
    cp -r --no-preserve=mode,ownership ${ast-grep-config} ./__aleph-lint-config
    trap 'rm -rf ./__aleph-lint-config' EXIT

    ${lib.getExe ast-grep} --config ./__aleph-lint-config/sgconfig.yml test --update-all
  '';

  text = ''
    cp -r --no-preserve=mode,ownership ${ast-grep-config} ./__aleph-lint-config
    trap 'rm -rf ./__aleph-lint-config' EXIT

    ${lib.getExe ast-grep} --config ./__aleph-lint-config/sgconfig.yml scan --context 2 --color always "$@"
  '';
}
