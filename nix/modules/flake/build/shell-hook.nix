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
  scripts-dir = ./scripts;

  # Buck2 prelude source (requires inputs.buck2-prelude if cfg.prelude.path not set)
  prelude-src = if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude;

  # Toolchains source (from this flake)
  toolchains-src = inputs.self + "/toolchains";

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      # Convert vars attrset to env var exports
      # Dhall expects UPPER_SNAKE_CASE env vars
      env-vars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          "nativeBuildInputs" = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # Prelude and toolchains linking
  prelude-hook =
    if isLinux && cfg.prelude.enable then
      render-dhall "shell-hook-prelude.bash" (scripts-dir + "/shell-hook-prelude.dhall") {
        "prelude_src" = prelude-src;
        "toolchains_src" = toolchains-src;
      }
    else
      null;

  # .buckconfig generation (main file, if enabled)
  buckconfig-main-hook =
    if isLinux && cfg.generate-buckconfig-main then
      let
        buckconfig-main-ini = pkgs.writeText "buckconfig-main.ini" (
          builtins.readFile (scripts-dir + "/buckconfig-main.ini")
        );
      in
      render-dhall "shell-hook-buckconfig-main.bash" (scripts-dir + "/shell-hook-buckconfig-main.dhall") {
        "buckconfig_main_ini" = buckconfig-main-ini;
      }
    else
      null;

  # .buckconfig.local generation
  buckconfig-local-hook =
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
  haskell-wrappers-hook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.haskell.enable then
      render-dhall "haskell-wrappers.bash" (scripts-dir + "/haskell-wrappers.dhall") {
        "scripts_dir" = scripts-dir;
      }
    else
      null;

  # Lean wrappers
  lean-wrappers-hook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.lean.enable then
      render-dhall "lean-wrappers.bash" (scripts-dir + "/lean-wrappers.dhall") {
        "scripts_dir" = scripts-dir;
      }
    else
      null;

  # C++ wrappers
  cxx-wrappers-hook =
    if isLinux && cfg.generate-wrappers && cfg.toolchain.cxx.enable then
      render-dhall "cxx-wrappers.bash" (scripts-dir + "/cxx-wrappers.dhall") {
        "scripts_dir" = scripts-dir;
      }
    else
      null;

  # Auto-generate compile_commands.json
  compdb-auto-hook =
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
  all-hooks = lib.filter (x: x != null) [
    prelude-hook
    buckconfig-main-hook
    # Wrappers BEFORE buckconfig.local (buckconfig references bin/ghc wrapper)
    haskell-wrappers-hook
    lean-wrappers-hook
    cxx-wrappers-hook
    # buckconfig.local AFTER wrappers
    buckconfig-local-hook
    compdb-auto-hook
  ];

  # Generate combined shell hook
  shell-hook =
    if all-hooks == [ ] then
      ""
    else
      let
        combined-hook = pkgs.runCommand "aleph-build-shell-hook.bash" { } ''
          echo "# aleph.build shell hook" > $out
          echo "mkdir -p bin" >> $out
          ${lib.concatMapStringsSep "\n" (hook: "cat ${hook} >> $out") all-hooks}
          echo 'echo "Generated bin/ wrappers for Buck2 toolchains"' >> $out
        '';
      in
      ''
        source ${combined-hook}
      '';
in
{
  "shellHook" = shell-hook;
}
