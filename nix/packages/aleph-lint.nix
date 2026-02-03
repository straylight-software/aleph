{
  pkgs,
  tree-sitter-grammars,
  tree-sitter,
  ast-grep,
  lib,
  dhall,
  runCommand,
}:
let
  inherit (pkgs.aleph) write-shell-application;
  linter-src = ../../linter;

  # Build the linter package with the Prelude as a dependency
  # The Prelude from nixpkgs already has proper caching
  linter-dhall-package = pkgs.dhallPackages.buildDhallDirectoryPackage {
    name = "aleph-linter";
    src = linter-src;
    file = "generate.dhall";
    dependencies = [ pkgs.dhallPackages.Prelude ];
    source = true;
  };

  # Generate the ast-grep config using the cached dhall package
  ast-grep-config = runCommand "ast-grep-config"
    {
      nativeBuildInputs = [ dhall ];
    }
    ''
      mkdir -p $out
      
      # Set up dhall cache from all dependencies
      export XDG_CACHE_HOME=$TMPDIR/.cache
      mkdir -p $XDG_CACHE_HOME/dhall
      
      # Copy the cached imports from the Prelude package
      if [ -d ${pkgs.dhallPackages.Prelude}/.cache/dhall ]; then
        cp -r ${pkgs.dhallPackages.Prelude}/.cache/dhall/* $XDG_CACHE_HOME/dhall/
      fi
      
      # Create working directory with linter files
      WORKDIR=$TMPDIR/linter
      mkdir -p $WORKDIR
      cp -r ${linter-src}/* $WORKDIR/
      chmod -R +w $WORKDIR
      
      # Generate the config using the cached dhall
      cd $WORKDIR
      dhall to-directory-tree --file ./generate.dhall --output $out
    '';
in
write-shell-application {
  name = "aleph-lint";
  runtime-inputs = [ ast-grep tree-sitter tree-sitter-grammars.tree-sitter-nix ];

  derivation-args.post-check = ''
    ${lib.getExe ast-grep} --config ${ast-grep-config}/sgconfig.yaml test || true
  '';

  text = ''
    # Copy the entire ast-grep config directory structure
    cp -r --no-preserve=mode ${ast-grep-config} ./__aleph-lint-config
    chmod -R +w ./__aleph-lint-config
    trap 'rm -rf ./__aleph-lint-config' EXIT

    ${lib.getExe ast-grep} --config ./__aleph-lint-config/sgconfig.yaml scan --context 2 --color always "$@"
  '';
}
