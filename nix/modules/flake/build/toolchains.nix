# nix/modules/flake/build/toolchains.nix
#
# Toolchain path resolution for Buck2
#
{
  lib,
  pkgs,
  cfg,
}:
let
  inherit (pkgs.stdenv) isLinux;

  # LLVM 22 from llvm-git overlay
  llvm-git = pkgs.llvm-git or null;

  # GCC for libstdc++ headers and runtime
  gcc = pkgs.gcc15 or pkgs.gcc14 or pkgs.gcc;
  gcc-unwrapped = gcc.cc;
  gcc-version = gcc-unwrapped.version;
  triple = pkgs.stdenv.hostPlatform.config;

  # NVIDIA SDK
  nvidia-sdk = pkgs.nvidia-sdk or null;

  # mdspan (Kokkos reference implementation)
  mdspan = pkgs.mdspan or null;

  # Haskell
  hsPkgs = pkgs.haskell.packages.ghc912 or pkgs.haskellPackages;
  ghcVersion = hsPkgs.ghc.version;

  # The Haskell package universe
  hsPackageList = cfg.toolchain.haskell.packages hsPkgs;

  # GHC with all packages baked in
  ghcForBuck2 = hsPkgs.ghcWithPackages (_: hsPackageList);

  # Extract package info for buckconfig.local
  hsPackageInfo =
    pkg:
    let
      name = pkg.pname or (builtins.parseDrvName pkg.name).name;
      version = pkg.version or "0";
      libdir = "${pkg}/lib/ghc-${ghcVersion}/lib";
      confDir = "${libdir}/package.conf.d";
      confFiles = builtins.attrNames (builtins.readDir confDir);
      confFile = lib.head (lib.filter (f: lib.hasSuffix ".conf" f) confFiles);
      id = lib.removeSuffix ".conf" confFile;
    in
    {
      inherit
        name
        version
        id
        libdir
        ;
      path = "${pkg}";
      db = confDir;
    };

  # Generate buckconfig entries for all packages
  hsPackagesConfig = lib.concatMapStringsSep "\n" (
    pkg:
    let
      info = hsPackageInfo pkg;
    in
    ''
      ${info.name} = ${info.path}
      ${info.name}.db = ${info.db}
      ${info.name}.libdir = ${info.libdir}
      ${info.name}.id = ${info.id}''
  ) hsPackageList;

  # Python
  python = pkgs.python312 or pkgs.python311;
  pythonEnv = python.withPackages cfg.toolchain.python.packages;

  # ────────────────────────────────────────────────────────────────────────────
  # Buck2 toolchain configuration attrset
  # ────────────────────────────────────────────────────────────────────────────
  buck2-toolchain =
    lib.optionalAttrs (isLinux && llvm-git != null) {
      # LLVM 22 compilers
      cc = "${llvm-git}/bin/clang";
      cxx = "${llvm-git}/bin/clang++";
      cpp = "${llvm-git}/bin/clang-cpp";

      # LLVM 22 bintools
      ar = "${llvm-git}/bin/llvm-ar";
      ld = "${llvm-git}/bin/ld.lld";
      nm = "${llvm-git}/bin/llvm-nm";
      objcopy = "${llvm-git}/bin/llvm-objcopy";
      objdump = "${llvm-git}/bin/llvm-objdump";
      strip = "${llvm-git}/bin/llvm-strip";
      ranlib = "${llvm-git}/bin/llvm-ranlib";

      # Include directories
      clang-resource-dir = "${llvm-git}/lib/clang/22";
      gcc-include = "${gcc-unwrapped}/include/c++/${gcc-version}";
      gcc-include-arch = "${gcc-unwrapped}/include/c++/${gcc-version}/${triple}";
      glibc-include = "${pkgs.glibc.dev}/include";

      # Library directories
      gcc-lib = "${gcc-unwrapped}/lib/gcc/${triple}/${gcc-version}";
      gcc-lib-base = "${gcc.cc.lib}/lib";
      glibc-lib = "${pkgs.glibc}/lib";
    }
    // lib.optionalAttrs (isLinux && mdspan != null) {
      mdspan-include = "${mdspan}/include";
    }
    // lib.optionalAttrs (isLinux && nvidia-sdk != null && cfg.toolchain.nv.enable) {
      nvidia-sdk-path = "${nvidia-sdk}";
      nvidia-sdk-include = "${nvidia-sdk}/include";
      nvidia-sdk-lib = "${nvidia-sdk}/lib";
      nv-archs = lib.concatStringsSep "," cfg.toolchain.nv.archs;
    }
    // lib.optionalAttrs (isLinux && cfg.toolchain.python.enable) {
      python-interpreter = "${pythonEnv}/bin/python3";
      python-include = "${python}/include/python3.12";
      python-lib = "${python}/lib";
      nanobind-include = "${python.pkgs.nanobind}/lib/python3.12/site-packages/nanobind/include";
      nanobind-cmake = "${python.pkgs.nanobind}/lib/python3.12/site-packages/nanobind";
      pybind11-include = "${python.pkgs.pybind11}/lib/python3.12/site-packages/pybind11/include";
    };

  # ────────────────────────────────────────────────────────────────────────────
  # Packages for devshell
  # ────────────────────────────────────────────────────────────────────────────
  packages =
    lib.optionals (isLinux && llvm-git != null && cfg.toolchain.cxx.enable) [ llvm-git ]
    ++ lib.optionals (isLinux && nvidia-sdk != null && cfg.toolchain.nv.enable) [ nvidia-sdk ]
    ++ lib.optionals cfg.toolchain.haskell.enable [ ghcForBuck2 ]
    ++ lib.optionals (cfg.toolchain.rust.enable && pkgs ? rustc) [
      pkgs.rustc
      pkgs.cargo
      pkgs.clippy
      pkgs.rustfmt
      pkgs.rust-analyzer
    ]
    ++ lib.optionals (cfg.toolchain.lean.enable && pkgs ? lean4) [ pkgs.lean4 ]
    ++ lib.optionals cfg.toolchain.python.enable [ pythonEnv ]
    ++ lib.optionals (pkgs ? buck2) [ pkgs.buck2 ];
in
{
  inherit
    buck2-toolchain
    packages
    ghcForBuck2
    ghcVersion
    hsPackagesConfig
    llvm-git
    nvidia-sdk
    pythonEnv
    ;
}
