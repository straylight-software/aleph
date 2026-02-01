{
  pkgs,
  tree-sitter-grammars,
  tree-sitter,
  ast-grep,
  writers,
  lib,
  dhall-yaml,
}:
let
  inherit (pkgs.aleph) write-shell-application;

  # Import the linter source directories into the nix store, including dhall files
  linter-src = lib.sourceFilesBySuffices ../../linter [ ".dhall" ];

  # Function to convert a directory of .dhall files to .yml files
  convertDhallDir =
    name: dhallDirSrc:
    pkgs.runCommand name
      {
        nativeBuildInputs = [ dhall-yaml ];
      }
      ''
        mkdir -p $out
        if [ -d "${dhallDirSrc}" ]; then
          for dhall_file in $(find ${dhallDirSrc} -maxdepth 1 -name "*.dhall" -type f | sort); do
            if [ -f "$dhall_file" ]; then
              filename=$(basename "$dhall_file" .dhall)
              echo "Converting: $dhall_file -> $out/$filename.yml"
              dhall-to-yaml-ng --file "$dhall_file" --output "$out/$filename.yml"
            fi
          done
        fi
      '';

  # Convert rules, tests, and utils from dhall to yaml
  rules-yaml = convertDhallDir "linter-rules-yaml" (linter-src + "/rules");
  tests-yaml = convertDhallDir "linter-tests-yaml" (linter-src + "/rule-tests");
  utils-yaml = convertDhallDir "linter-utils-yaml" (linter-src + "/utils");

  sgconfig = {
    "ruleDirs" = [ "${rules-yaml}" ];
    "testConfigs" = [
      { "testDir" = "${tests-yaml}"; }
    ];
    "utilDirs" = [ "${utils-yaml}" ];
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
