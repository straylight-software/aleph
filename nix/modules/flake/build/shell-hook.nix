# nix/modules/flake/build/shell-hook.nix
#
# Shell hook for Buck2 setup (prelude linking, buckconfig generation, wrappers)
#
{
  lib,
  pkgs,
  cfg,
  inputs,
  buckconfig,
}:
let
  inherit (pkgs.stdenv) isLinux;
  scriptsDir = ./scripts;

  # Buck2 prelude source
  preludeSrc = if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude or null;

  # Toolchains source (from this flake)
  toolchainsSrc = inputs.self + "/toolchains";

  # Substitute @var@ placeholders in script files
  substituteScript =
    file: vars:
    let
      substitutions = lib.concatStringsSep " " (
        lib.mapAttrsToList (name: value: "--subst-var-by ${name} '${toString value}'") vars
      );
    in
    pkgs.runCommand (baseNameOf file) { } ''
      substitute ${file} $out ${substitutions}
    '';

  # Prelude and toolchains linking
  preludeHook =
    if isLinux && cfg.prelude.enable && preludeSrc != null then
      substituteScript (scriptsDir + "/shell-hook-prelude.bash") {
        preludeSrc = preludeSrc;
        toolchainsSrc = toolchainsSrc;
      }
    else
      null;

  # .buckconfig generation (main file, if enabled)
  buckconfigMainHook =
    if isLinux && cfg.generate-buckconfig-main then
      substituteScript (scriptsDir + "/shell-hook-buckconfig-main.bash") {
        buckconfigMain = builtins.readFile (scriptsDir + "/buckconfig-main.ini");
      }
    else
      null;

  # .buckconfig.local generation
  buckconfigLocalHook =
    if isLinux && cfg.generate-buckconfig then
      pkgs.writeText "buckconfig-local-hook.bash" ''
        # Generate .buckconfig.local with Nix store paths
        rm -f .buckconfig.local 2>/dev/null || true
        cp ${buckconfig.buckconfig-local} .buckconfig.local
        chmod 644 .buckconfig.local
        echo "Generated .buckconfig.local with Nix store paths"
      ''
    else
      null;

  # Haskell wrappers
  haskellWrappersHook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.haskell.enable then
      substituteScript (scriptsDir + "/haskell-wrappers.bash") {
        scriptsDir = scriptsDir;
      }
    else
      null;

  # Lean wrappers
  leanWrappersHook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.lean.enable then
      substituteScript (scriptsDir + "/lean-wrappers.bash") {
        scriptsDir = scriptsDir;
      }
    else
      null;

  # C++ wrappers
  cxxWrappersHook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.cxx.enable then
      substituteScript (scriptsDir + "/cxx-wrappers.bash") {
        scriptsDir = scriptsDir;
      }
    else
      null;

  # Auto-generate compile_commands.json
  compdbAutoHook =
    if isLinux && cfg.toolchain.cxx.enable && cfg.compdb.enable && cfg.compdb.auto-generate then
      let
        targets = lib.concatStringsSep " " cfg.compdb.targets;
      in
      pkgs.writeText "compdb-auto-hook.bash" ''
        # Auto-generate compile_commands.json for clangd
        if command -v buck2 &>/dev/null; then
          echo "Generating compile_commands.json..."
          COMPDB_PATH=$(buck2 bxl prelude//cxx/tools/compilation_database.bxl:generate -- --targets ${targets} 2>/dev/null | tail -1) || true
          if [ -n "$COMPDB_PATH" ] && [ -f "$COMPDB_PATH" ]; then
            cp "$COMPDB_PATH" compile_commands.json
            echo "Generated compile_commands.json ($(jq length compile_commands.json 2>/dev/null || echo '?') entries)"
          fi
        fi
      ''
    else
      null;

  # Combine all hooks into a single script
  # Order matters: wrappers must be created before buckconfig.local references them
  allHooks = lib.filter (x: x != null) [
    preludeHook
    buckconfigMainHook
    # Wrappers BEFORE buckconfig.local (buckconfig references bin/ghc wrapper)
    haskellWrappersHook
    leanWrappersHook
    cxxWrappersHook
    # buckconfig.local AFTER wrappers
    buckconfigLocalHook
    compdbAutoHook
  ];

  # Generate combined shell hook
  shellHook =
    if allHooks == [ ] then
      ""
    else
      let
        combinedHook = pkgs.runCommand "aleph-build-shell-hook.bash" { } ''
          echo "# aleph.build shell hook" > $out
          echo "mkdir -p bin" >> $out
          ${lib.concatMapStringsSep "\n" (hook: "cat ${hook} >> $out") allHooks}
          echo 'echo "Generated bin/ wrappers for Buck2 toolchains"' >> $out
        '';
      in
      ''
        source ${combinedHook}
      '';
in
{
  inherit shellHook;
}
