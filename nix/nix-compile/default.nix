# nix/nix-compile/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // nix-compile.nix //
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
#   aleph.nix-compile.cli        CLI tool: nix-compile parse|infer|check
#   aleph.nix-compile.parse      Parse script, return schema as Nix attrset
#   aleph.nix-compile.check      Check script for policy violations
#   aleph.nix-compile.shell      Development shell with nix-compile tools
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, pkgs }:
let
  inherit (lib) optionalString concatStringsSep;
  inherit (builtins) isString isAttrs;

  # Source directories
  # :: Path
  # :: Path
  nix-compile-lib = ./lib;
  nix-compile-app = ./app;
# :: t11

  # Use the script GHC (already has ShellCheck, hnix)
  ghc = pkgs.aleph.script.ghc;

  # ────────────────────────────────────────────────────────────────────────────
  # // CLI tool //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # nix-compile parse <script>   Parse and show extracted facts
  # :: t13
  # :: "nix-compile"
  # :: [Path]
  # :: String
  # nix-compile infer <script>   Infer types and show schema (JSON)
  # nix-compile check <script>   Check for policy violations

  cli = pkgs.writeShellApplication {
    name = "nix-compile";
    runtimeInputs = [ ghc ];
    text = ''
      exec runghc -i${nix-compile-lib} ${nix-compile-app}/nix-compile.hs "$@"
    '';
  };
# :: t15
# :: "nix-compile"
# :: Path
# :: Bool
# :: [Path]
# :: String

  # ────────────────────────────────────────────────────────────────────────────
  # // compiled CLI //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Compiled version for faster execution in CI

  # :: String
  compiled = pkgs.stdenv.mkDerivation {
    name = "nix-compile";
    src = ./.;
    dontUnpack = true;
    nativeBuildInputs = [ ghc ];
    buildPhase = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports \
        -hidir . -odir . \
        -i${nix-compile-lib} \
        -o nix-compile ${nix-compile-app}/nix-compile.hs
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin
      cp nix-compile $out/bin/
      runHook postInstall
    '';
  };

  # :: t16 -> t24
  # ────────────────────────────────────────────────────────────────────────────
  # // Nix integration //
  # :: [Path]
  # :: t21
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Parse a script and return its schema as a Nix attrset.
  # This runs at eval time via IFD (import from derivation).
  #
  # Usage:
  #   schema = nix-compile.parse ./deploy.sh;
  #   assert schema.env.PORT.type == "TInt";
  #
  # NOTE: IFD is disabled in pure evaluation mode and CI.
  # For CI, use the check derivation below instead.

  parse =
    scriptPath:
    # :: t25 -> t29
    let
      # :: [Path]
      result = pkgs.runCommand "nix-compile-schema" { nativeBuildInputs = [ cli ]; } ''
        nix-compile infer ${scriptPath} > $out
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
  #   checks.deploy = nix-compile.check ./deploy.sh;

  check =
    scriptPath:
    pkgs.runCommand "nix-compile-check-${builtins.baseNameOf scriptPath}" { nativeBuildInputs = [ cli ]; } ''
      nix-compile check ${scriptPath}
      touch $out
    '';

  # ────────────────────────────────────────────────────────────────────────────
  # // Script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Build a shell script with nix-compile.nix analysis.
  #
  # Usage:
  #   # Simple form
  #   pkgs.nix-compile.mkScript "my-script" ''
  #     PORT="''${PORT:-8080}"
  #     config.server.port=$PORT
  #     exec ${pkgs.myapp}/bin/myapp --config <(emit-config)
  #   ''
  #
  #   # Full form
  #   pkgs.nix-compile.mkScript {
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
          throw "nix-compile.mkScript: expected (name, script) or { name, script, ... }";

      name = args.name or (throw "nix-compile.mkScript: 'name' is required");
      script = args.script or (throw "nix-compile.mkScript: 'script' is required");
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
            nix-compile emit ${scriptFile} > $out
          '';

      # Check for policy violations (bare commands)
      policyCheck =
        pkgs.runCommand "${name}-policy-check"
          {
            nativeBuildInputs = [ cli ];
          }
          ''
            nix-compile check ${scriptFile}
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
    name = "nix-compile-shell";
    buildInputs = [
      ghc
      cli
      pkgs.jq # JSON pretty-printing
    ];
    shellHook = ''
      # :: { app : t3, lib : t2 }
      # :: t2
      # :: t3
      echo "nix-compile.nix development shell"
      echo "  nix-compile parse <script>   Show facts"
      echo "  nix-compile infer <script>   Show schema (JSON)"
      echo "  nix-compile check <script>   Check policies"
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
    lib = nix-compile-lib;
    app = nix-compile-app;
  };
}
