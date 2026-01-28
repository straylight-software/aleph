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
    # Use unique config file per invocation to avoid race conditions
    # when treefmt runs multiple aleph-lint instances in parallel
    SGCONFIG_TMP="$(mktemp -t aleph-lint-XXXXXX.yml)"
    cp --no-preserve=mode ${sgconfig-yml} "$SGCONFIG_TMP"
    trap 'rm -f "$SGCONFIG_TMP"' EXIT

    ${lib.getExe ast-grep} \
      --config "$SGCONFIG_TMP" \
      scan \
      --context 2 \
      --color always \
      "$@"
  '';
}
