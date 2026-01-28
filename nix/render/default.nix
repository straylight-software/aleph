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
  inherit (lib) optionalString concatStringsSep;
  inherit (builtins) isString isAttrs;

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
  # // Script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Build a shell script with render.nix analysis.
  #
  # Usage:
  #   # Simple form
  #   pkgs.render.mkScript "my-script" ''
  #     PORT="''${PORT:-8080}"
  #     config.server.port=$PORT
  #     exec ${pkgs.myapp}/bin/myapp --config <(emit-config)
  #   ''
  #
  #   # Full form
  #   pkgs.render.mkScript {
  #     name = "my-script";
  #     script = ''...'';
  #     deps = [ pkgs.jq ];           # Added to PATH
  #     requireStorePaths = true;     # Fail on bare commands (default: true)
  #     injectEmitConfig = true;      # Add emit-config function (default: true)
  #   }

  mkScript =
    arg1: arg2:
    let
      # Parse arguments: mkScript "name" "script" or mkScript { ... }
      args =
        if isString arg1 && isString arg2 then
          {
            name = arg1;
            script = arg2;
          }
        else if isAttrs arg1 && arg2 == null then
          arg1
        else if isAttrs arg1 then
          arg1
        else
          throw "render.mkScript: expected (name, script) or { name, script, ... }";

      name = args.name or (throw "render.mkScript: 'name' is required");
      script = args.script or (throw "render.mkScript: 'script' is required");
      deps = args.deps or [ ];
      requireStorePaths = args.requireStorePaths or true;
      injectEmitConfig = args.injectEmitConfig or true;

      # Write script to a file for analysis
      scriptFile = pkgs.writeText "${name}-source.sh" script;

      # Analyze the script and get emit-config function
      emitConfigFunc =
        pkgs.runCommand "${name}-emit-config"
          {
            nativeBuildInputs = [ cli ];
          }
          ''
            render emit ${scriptFile} > $out
          '';

      # Check for policy violations (bare commands)
      policyCheck =
        pkgs.runCommand "${name}-policy-check"
          {
            nativeBuildInputs = [ cli ];
          }
          ''
            render check ${scriptFile}
            touch $out
          '';

      # Build the final script
      finalScript =
        let
          # Inject emit-config at the start (after shebang/set lines)
          emitConfigContent = if injectEmitConfig then builtins.readFile emitConfigFunc else "";

          # PATH setup for deps
          pathSetup = if deps != [ ] then "export PATH=\"${lib.makeBinPath deps}\${PATH:+:$PATH}\"" else "";
        in
        ''
          ${pathSetup}
          ${optionalString injectEmitConfig emitConfigContent}
          ${script}
        '';

    in
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = deps;
      text = script;
      derivationArgs = {
        # Include emit-config in passthru
        passthru = {
          # The analyzed schema (IFD)
          schema = parse scriptFile;
          # The emit-config function source
          emitConfig = emitConfigFunc;
          # For debugging
          sourceScript = scriptFile;
        };
        # Policy check as a dependency (fails build if violations)
        nativeBuildInputs = lib.optional requireStorePaths policyCheck;
      };
    };

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
    mkScript
    shell
    ;

  # Convenience: passthru source paths
  src = {
    lib = render-lib;
    app = render-app;
  };
}
