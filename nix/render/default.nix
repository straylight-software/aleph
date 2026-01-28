# nix/render/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // render.nix //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games. Cyberspace. A
#     consensual hallucination experienced daily by billions of legitimate
#     operators, in every nation.
#
#                                                         — Neuromancer
#
# Type inference for bash scripts at Nix eval time.
#
# Eliminates runtime bash bugs by catching type errors, missing variables,
# and policy violations during the build.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   aleph.render.cli        CLI tool: render parse|infer|check
#   aleph.render.parse      Parse script, return schema as Nix attrset
#   aleph.render.check      Check script for policy violations
#   aleph.render.shell      Development shell with render tools
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, pkgs }:
let
  # Source directories
  render-lib = ./lib;
  render-app = ./app;

  # Use the script GHC (already has ShellCheck, hnix)
  ghc = pkgs.aleph.script.ghc;

  # ────────────────────────────────────────────────────────────────────────────
  # // CLI tool //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # render parse <script>   Parse and show extracted facts
  # render infer <script>   Infer types and show schema (JSON)
  # render check <script>   Check for policy violations

  cli = pkgs.writeShellApplication {
    name = "render";
    runtimeInputs = [ ghc ];
    text = ''
      exec runghc -i${render-lib} ${render-app}/render.hs "$@"
    '';
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // compiled CLI //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Compiled version for faster execution in CI

  compiled = pkgs.stdenv.mkDerivation {
    name = "render";
    src = ./.;
    dontUnpack = true;
    nativeBuildInputs = [ ghc ];
    buildPhase = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports \
        -hidir . -odir . \
        -i${render-lib} \
        -o render ${render-app}/render.hs
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin
      cp render $out/bin/
      runHook postInstall
    '';
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // Nix integration //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Parse a script and return its schema as a Nix attrset.
  # This runs at eval time via IFD (import from derivation).
  #
  # Usage:
  #   schema = render.parse ./deploy.sh;
  #   assert schema.env.PORT.type == "TInt";
  #
  # NOTE: IFD is disabled in pure evaluation mode and CI.
  # For CI, use the check derivation below instead.

  parse =
    scriptPath:
    let
      result = pkgs.runCommand "render-schema" { nativeBuildInputs = [ cli ]; } ''
        render infer ${scriptPath} > $out
      '';
    in
    builtins.fromJSON (builtins.readFile result);

  # ────────────────────────────────────────────────────────────────────────────
  # // Build-time check //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Creates a derivation that fails if the script has policy violations.
  # Use this in CI checks.
  #
  # Usage:
  #   checks.deploy = render.check ./deploy.sh;

  check =
    scriptPath:
    pkgs.runCommand "render-check-${builtins.baseNameOf scriptPath}" { nativeBuildInputs = [ cli ]; } ''
      render check ${scriptPath}
      touch $out
    '';

  # ────────────────────────────────────────────────────────────────────────────
  # // Shell //
  # ────────────────────────────────────────────────────────────────────────────

  shell = pkgs.mkShell {
    name = "render-shell";
    buildInputs = [
      ghc
      cli
      pkgs.jq # JSON pretty-printing
    ];
    shellHook = ''
      echo "render.nix development shell"
      echo "  render parse <script>   Show facts"
      echo "  render infer <script>   Show schema (JSON)"
      echo "  render check <script>   Check policies"
    '';
  };

in
{
  inherit
    cli
    compiled
    parse
    check
    shell
    ;

  # Convenience: passthru source paths
  src = {
    lib = render-lib;
    app = render-app;
  };
}
