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
    ghc-for-buck2
    ghc-version
    hs-packages-config
    ;

  # Lisp-case aliases for lib functions
  concat-map-strings-sep = lib.concatMapStringsSep;
  concat-strings-sep = lib.concatStringsSep;
  map-attrs' = lib.mapAttrs';
  name-value-pair = lib.nameValuePair;
  replace-strings = builtins.replaceStrings;
  to-upper = lib.toUpper;

  scripts-dir = ./scripts;

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      # Convert vars attrset to env var exports
      # Dhall expects UPPER_SNAKE_CASE env vars
      env-vars = map-attrs' (
        k: v: name-value-pair (to-upper (replace-strings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # Build config sections from Dhall templates
  cxx-config =
    if cfg.toolchain.cxx.enable && buck2-toolchain ? cc then
      render-dhall "buckconfig-cxx.ini" (scripts-dir + "/buckconfig-cxx.dhall") {
        inherit (buck2-toolchain) cc;
        inherit (buck2-toolchain) cxx;
        inherit (buck2-toolchain) cpp;
        inherit (buck2-toolchain) ar;
        inherit (buck2-toolchain) ld;
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

  # Turing Registry flags
  flags-config =
    if cfg.toolchain.cxx.enable && buck2-toolchain ? c-flags then
      render-dhall "buckconfig-flags.ini" (scripts-dir + "/buckconfig-flags.dhall") {
        c_flags = concat-strings-sep " " buck2-toolchain.c-flags;
        cxx_flags = concat-strings-sep " " buck2-toolchain.cxx-flags;
      }
    else
      null;

  haskell-config =
    if cfg.toolchain.haskell.enable then
      render-dhall "buckconfig-haskell.ini" (scripts-dir + "/buckconfig-haskell.dhall") {
        # Use bin/ghc wrapper which:
        # 1. Filters Mercury-specific flags
        # 2. Resolves -package to -package-id (GHC 9.12 workaround)
        ghc = "bin/ghc";
        ghc_pkg = "bin/ghc-pkg";
        haddock = "bin/haddock";
        ghc_version = ghc-version;
        ghc_lib_dir = "${ghc-for-buck2}/lib/ghc-${ghc-version}/lib";
        global_package_db = "${ghc-for-buck2}/lib/ghc-${ghc-version}/lib/package.conf.d";
      }
    else
      null;

  python-config =
    if cfg.toolchain.python.enable && buck2-toolchain ? python-interpreter then
      render-dhall "buckconfig-python.ini" (scripts-dir + "/buckconfig-python.dhall") {
        interpreter = buck2-toolchain.python-interpreter;
        python_include = buck2-toolchain.python-include;
        python_lib = buck2-toolchain.python-lib;
        nanobind_include = buck2-toolchain.nanobind-include;
        nanobind_cmake = buck2-toolchain.nanobind-cmake;
        pybind11_include = buck2-toolchain.pybind11-include;
      }
    else
      null;

  nv-config =
    if cfg.toolchain.nv.enable && buck2-toolchain ? nvidia-sdk-path then
      render-dhall "buckconfig-nv.ini" (scripts-dir + "/buckconfig-nv.dhall") {
        nvidia_sdk_path = buck2-toolchain.nvidia-sdk-path;
        nvidia_sdk_include = buck2-toolchain.nvidia-sdk-include;
        nvidia_sdk_lib = buck2-toolchain.nvidia-sdk-lib;
        archs = buck2-toolchain.nv-archs;
      }
    else
      null;

  rust-config =
    if cfg.toolchain.rust.enable && pkgs ? rustc then
      render-dhall "buckconfig-rust.ini" (scripts-dir + "/buckconfig-rust.dhall") {
        rustc = "${pkgs.rustc}/bin/rustc";
        rustdoc = "${pkgs.rustc}/bin/rustdoc";
        clippy_driver = "${pkgs.clippy}/bin/clippy-driver";
        cargo = "${pkgs.cargo}/bin/cargo";
      }
    else
      null;

  lean-config =
    if cfg.toolchain.lean.enable && pkgs ? lean4 then
      render-dhall "buckconfig-lean.ini" (scripts-dir + "/buckconfig-lean.dhall") {
        lean = "${pkgs.lean4}/bin/lean";
        leanc = "${pkgs.lean4}/bin/leanc";
        lake = "${pkgs.lean4}/bin/lake";
        lean_lib_dir = "${pkgs.lean4}/lib/lean/library";
        lean_include_dir = "${pkgs.lean4}/include";
      }
    else
      null;

  # Combine all config sections
  config-parts = lib.filter (x: x != null) [
    cxx-config
    flags-config
    haskell-config
    python-config
    nv-config
    rust-config
    lean-config
  ];

  # Haskell packages config as a separate file
  haskell-packages-file =
    if cfg.toolchain.haskell.enable then
      pkgs.writeText "haskell-packages.ini" ''

        [haskell.packages]
        ${hs-packages-config}
      ''
    else
      null;

  # Generate the final .buckconfig.local by concatenating all parts
  all-config-parts =
    config-parts ++ lib.optional (haskell-packages-file != null) haskell-packages-file;

  buckconfig-local = pkgs.runCommand "buckconfig.local" { } ''
    cat ${concat-map-strings-sep " " toString all-config-parts} > $out
  '';
in
{
  inherit buckconfig-local scripts-dir;

  # For use in shell-hook - use Dhall rendering
  render-dhall =
    name: src: vars:
    render-dhall name src vars;
}
