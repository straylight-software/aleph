# nix/modules/flake/build/buckconfig.nix
#
# .buckconfig.local generation using Dhall templates
#
{
  lib,
  pkgs,
  cfg,
  toolchains,
}:
let
  inherit (toolchains)
    buck2-toolchain
    ghcForBuck2
    ghcVersion
    hsPackagesConfig
    ;

  scriptsDir = ./scripts;

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

  # Build config sections from Dhall templates
  cxxConfig =
    if cfg.toolchain.cxx.enable && buck2-toolchain ? cc then
      renderDhall "buckconfig-cxx.ini" (scriptsDir + "/buckconfig-cxx.dhall") {
        cc = buck2-toolchain.cc;
        cxx = buck2-toolchain.cxx;
        cpp = buck2-toolchain.cpp;
        ar = buck2-toolchain.ar;
        ld = buck2-toolchain.ld;
        clang_resource_dir = buck2-toolchain.clang-resource-dir;
        gcc_include = buck2-toolchain.gcc-include;
        gcc_include_arch = buck2-toolchain.gcc-include-arch;
        glibc_include = buck2-toolchain.glibc-include;
        gcc_lib = buck2-toolchain.gcc-lib;
        gcc_lib_base = buck2-toolchain.gcc-lib-base;
        glibc_lib = buck2-toolchain.glibc-lib;
        mdspan_include = buck2-toolchain.mdspan-include or "";
      }
    else
      null;

  haskellConfig =
    if cfg.toolchain.haskell.enable then
      renderDhall "buckconfig-haskell.ini" (scriptsDir + "/buckconfig-haskell.dhall") {
        # Use bin/ghc wrapper which:
        # 1. Filters Mercury-specific flags
        # 2. Resolves -package to -package-id (GHC 9.12 workaround)
        ghc = "bin/ghc";
        ghc_pkg = "bin/ghc-pkg";
        haddock = "bin/haddock";
        ghc_version = ghcVersion;
        ghc_lib_dir = "${ghcForBuck2}/lib/ghc-${ghcVersion}/lib";
        global_package_db = "${ghcForBuck2}/lib/ghc-${ghcVersion}/lib/package.conf.d";
      }
    else
      null;

  pythonConfig =
    if cfg.toolchain.python.enable && buck2-toolchain ? python-interpreter then
      renderDhall "buckconfig-python.ini" (scriptsDir + "/buckconfig-python.dhall") {
        interpreter = buck2-toolchain.python-interpreter;
        python_include = buck2-toolchain.python-include;
        python_lib = buck2-toolchain.python-lib;
        nanobind_include = buck2-toolchain.nanobind-include;
        nanobind_cmake = buck2-toolchain.nanobind-cmake;
        pybind11_include = buck2-toolchain.pybind11-include;
      }
    else
      null;

  nvConfig =
    if cfg.toolchain.nv.enable && buck2-toolchain ? nvidia-sdk-path then
      renderDhall "buckconfig-nv.ini" (scriptsDir + "/buckconfig-nv.dhall") {
        nvidia_sdk_path = buck2-toolchain.nvidia-sdk-path;
        nvidia_sdk_include = buck2-toolchain.nvidia-sdk-include;
        nvidia_sdk_lib = buck2-toolchain.nvidia-sdk-lib;
        archs = buck2-toolchain.nv-archs;
      }
    else
      null;

  rustConfig =
    if cfg.toolchain.rust.enable && pkgs ? rustc then
      renderDhall "buckconfig-rust.ini" (scriptsDir + "/buckconfig-rust.dhall") {
        rustc = "${pkgs.rustc}/bin/rustc";
        rustdoc = "${pkgs.rustc}/bin/rustdoc";
        clippy_driver = "${pkgs.clippy}/bin/clippy-driver";
        cargo = "${pkgs.cargo}/bin/cargo";
      }
    else
      null;

  leanConfig =
    if cfg.toolchain.lean.enable && pkgs ? lean4 then
      renderDhall "buckconfig-lean.ini" (scriptsDir + "/buckconfig-lean.dhall") {
        lean = "${pkgs.lean4}/bin/lean";
        leanc = "${pkgs.lean4}/bin/leanc";
        lake = "${pkgs.lean4}/bin/lake";
        lean_lib_dir = "${pkgs.lean4}/lib/lean/library";
        lean_include_dir = "${pkgs.lean4}/include";
      }
    else
      null;

  # Combine all config sections
  configParts = lib.filter (x: x != null) [
    cxxConfig
    haskellConfig
    pythonConfig
    nvConfig
    rustConfig
    leanConfig
  ];

  # Haskell packages config as a separate file
  haskellPackagesFile =
    if cfg.toolchain.haskell.enable then
      pkgs.writeText "haskell-packages.ini" ''

        [haskell.packages]
        ${hsPackagesConfig}
      ''
    else
      null;

  # Generate the final .buckconfig.local by concatenating all parts
  allConfigParts = configParts ++ lib.optional (haskellPackagesFile != null) haskellPackagesFile;

  buckconfig-local = pkgs.runCommand "buckconfig.local" { } ''
    cat ${lib.concatMapStringsSep " " toString allConfigParts} > $out
  '';
in
{
  inherit buckconfig-local scriptsDir;

  # For use in shell-hook - use Dhall rendering
  renderDhall =
    name: src: vars:
    renderDhall name src vars;
}
