# nix/modules/flake/build/shell-hook.nix
#
# Shell hook for Buck2 setup (prelude linking, buckconfig generation, wrappers)
# Uses Dhall templates for type-safe variable substitution
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

  # Buck2 prelude source (requires inputs.buck2-prelude if cfg.prelude.path not set)
  preludeSrc = if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude;

  # Toolchains source (from this flake)
  toolchainsSrc = inputs.self + "/toolchains";

  # Render Dhall template with environment variables
  renderDhall =
    name: src: vars:
    let
      # Convert vars attrset to env var exports
      # Dhall expects UPPER_SNAKE_CASE env vars
      envVars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // envVars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # Prelude and toolchains linking
  preludeHook =
    if isLinux && cfg.prelude.enable then
      renderDhall "shell-hook-prelude.bash" (scriptsDir + "/shell-hook-prelude.dhall") {
        prelude_src = preludeSrc;
        toolchains_src = toolchainsSrc;
      }
    else
      null;

  # .buckconfig generation (main file, if enabled)
  buckconfigMainHook =
    if isLinux && cfg.generate-buckconfig-main then
      let
        buckconfigMainIni = pkgs.writeText "buckconfig-main.ini" (
          builtins.readFile (scriptsDir + "/buckconfig-main.ini")
        );
      in
      renderDhall "shell-hook-buckconfig-main.bash" (scriptsDir + "/shell-hook-buckconfig-main.dhall") {
        buckconfig_main_ini = buckconfigMainIni;
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
      renderDhall "haskell-wrappers.bash" (scriptsDir + "/haskell-wrappers.dhall") {
        scripts_dir = scriptsDir;
      }
    else
      null;

  # Lean wrappers
  leanWrappersHook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.lean.enable then
      renderDhall "lean-wrappers.bash" (scriptsDir + "/lean-wrappers.dhall") {
        scripts_dir = scriptsDir;
      }
    else
      null;

  # C++ wrappers
  cxxWrappersHook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.cxx.enable then
      renderDhall "cxx-wrappers.bash" (scriptsDir + "/cxx-wrappers.dhall") {
        scripts_dir = scriptsDir;
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
