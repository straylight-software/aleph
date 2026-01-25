{
  pkgs,
  tree-sitter-grammars,
  tree-sitter,
  ast-grep,
  writers,
  lib,
}:
let
  inherit (pkgs.aleph) write-shell-application;

  linter-src = ../../linter;

  sgconfig = {
    "ruleDirs" = [ "${linter-src}/rules" ];
    "testConfigs" = [
      { "testDir" = "${linter-src}/rule-tests"; }
    ];
    "utilDirs" = [ "${linter-src}/utils" ];
  };

  sgconfig-yml = writers.writeYAML "sgconfig.yaml" sgconfig;
in
write-shell-application {
  name = "aleph-lint";
  runtime-inputs = [
    ast-grep
    tree-sitter
    tree-sitter-grammars.tree-sitter-nix
  ];
  derivation-args.post-check = ''
    echo "Checking config ${sgconfig-yml}"

    ${lib.getExe ast-grep} \
      --config ${sgconfig-yml} \
      test
  '';
  text = ''
    cp --no-preserve=mode --force ${sgconfig-yml} ./__sgconfig.yml
    trap 'rm -f ./__sgconfig.yml' EXIT

    ${lib.getExe ast-grep} \
      --config ./__sgconfig.yml \
      scan \
      --context 2 \
      --color always \
      "$@"
  '';
}
