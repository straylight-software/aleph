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
with lib;
writeShellApplication {
  name = "wsn-lint";
  runtimeInputs = [
    ast-grep
    tree-sitter
    tree-sitter-grammars.tree-sitter-nix
  ];
  text = ''
    cp --no-preserve=mode --force ${sgconfigYml} ./__sgconfig.yml
    ${getExe ast-grep} --config ./__sgconfig.yml scan "$@" || true
    rm ./__sgconfig.yml
  '';
}
