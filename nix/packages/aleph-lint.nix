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
  text =
    builtins.replaceStrings
      [ "$SGCONFIG_YML" "$AST_GREP_BIN" ]
      [ "${sgconfig-yml}" "${lib.getExe ast-grep}" ]
      (builtins.readFile ../../linter/aleph-lint.bash);
}
