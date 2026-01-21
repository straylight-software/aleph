{
  tree-sitter-grammars,
  tree-sitter,
  ast-grep,
  writers,
  writeShellApplication,
  lib,
}:
let
  linterSrc = ../../linter;

  sgconfig = {
    ruleDirs = [ "${linterSrc}/rules" ];
    testConfigs = [
      { testDir = "${linterSrc}/rule-tests"; }
    ];
    utilDirs = [ "${linterSrc}/utils" ];
  };

  sgconfigYml = writers.writeYAML "sgconfig.yaml" sgconfig;
in
writeShellApplication {
  name = "wsn-lint";
  runtimeInputs = [
    ast-grep
    tree-sitter
    tree-sitter-grammars.tree-sitter-nix
  ];
  derivationArgs.postCheck = ''
    echo "Checking config ${sgconfigYml}"

    ${lib.getExe ast-grep} \
      --config ${sgconfigYml} \
      test
  '';
  text = ''
    cp --no-preserve=mode --force ${sgconfigYml} ./__sgconfig.yml
    trap 'rm -f ./__sgconfig.yml' EXIT

    ${lib.getExe ast-grep} \
      --config ./__sgconfig.yml \
      scan \
      --context 2 \
      --color always \
      "$@"
  '';
}
