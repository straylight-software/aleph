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

  # Turing Registry - authoritative build flags
  turing-registry =
    pkgs.aleph.turing-registry or {
      cflags = [ ];
      cxxflags = [ ];
    };

  # LLVM 22 from llvm-git overlay (added by flake-module.nix)
  llvm-git = pkgs.llvm-git;

  # GCC for libstdc++ headers and runtime
  gcc = pkgs.gcc15 or pkgs.gcc14 or pkgs.gcc;
  gcc-unwrapped = gcc.cc;
  gcc-version = gcc-unwrapped.version;
  triple = pkgs.stdenv.hostPlatform.config;

  # NVIDIA SDK (added by flake-module.nix)
  nvidia-sdk = pkgs.nvidia-sdk;

  # mdspan (Kokkos reference implementation, added by flake-module.nix)
  mdspan = pkgs.mdspan;

  # Haskell
  hs-pkgs = pkgs.haskell.packages.ghc912 or pkgs.haskellPackages;
  ghc-version = hs-pkgs.ghc.version;

  # The Haskell package universe
  hs-package-list = cfg.toolchain.haskell.packages hs-pkgs;

  # GHC with all packages baked in
  ghc-for-buck2 = hs-pkgs.ghcWithPackages (_: hs-package-list);

  # Extract package info for buckconfig.local
  hs-package-info =
    pkg:
    let
      name = pkg.pname or (builtins.parseDrvName pkg.name).name;
      version = pkg.version or "0";
      lib-dir = "${pkg}/lib/ghc-${ghc-version}/lib";
      conf-dir = "${lib-dir}/package.conf.d";
      conf-files = builtins.attrNames (builtins.readDir conf-dir);
      conf-file = lib.head (lib.filter (f: lib.hasSuffix ".conf" f) conf-files);
      id = lib.removeSuffix ".conf" conf-file;
    in
    {
      inherit
        name
        version
        id
        lib-dir
        ;
      path = "${pkg}";
      db = conf-dir;
    };

  # Generate buckconfig entries for all packages
  hs-packages-config = lib.concatMapStringsSep "\n" (
    pkg:
    let
      info = hs-package-info pkg;
    in
    ''
      ${info.name} = ${info.path}
      ${info.name}.db = ${info.db}
      ${info.name}.libdir = ${info.lib-dir}
      ${info.name}.id = ${info.id}''
  ) hs-package-list;

  # Python
  python = pkgs.python312 or pkgs.python311;
  python-env = python.withPackages cfg.toolchain.python.packages;

  # ────────────────────────────────────────────────────────────────────────────
  # Buck2 toolchain configuration attrset
  # ────────────────────────────────────────────────────────────────────────────
  buck2-toolchain =
    lib.optionalAttrs isLinux {
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

      # mdspan
      mdspan-include = "${mdspan}/include";

      # Turing Registry flags (the true names)
      c-flags = turing-registry.cflags;
      cxx-flags = turing-registry.cxxflags;
    }
    // lib.optionalAttrs (isLinux && cfg.toolchain.nv.enable) {
      nvidia-sdk-path = "${nvidia-sdk}";
      nvidia-sdk-include = "${nvidia-sdk}/include";
      nvidia-sdk-lib = "${nvidia-sdk}/lib";
      nv-archs = lib.concatStringsSep "," cfg.toolchain.nv.archs;
    }
    // lib.optionalAttrs (isLinux && cfg.toolchain.python.enable) {
      python-interpreter = "${python-env}/bin/python3";
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
    lib.optionals (isLinux && cfg.toolchain.cxx.enable) [ llvm-git ]
    ++ lib.optionals (isLinux && cfg.toolchain.nv.enable) [ nvidia-sdk ]
    ++ lib.optionals cfg.toolchain.haskell.enable [ ghc-for-buck2 ]
    ++ lib.optionals (cfg.toolchain.rust.enable && pkgs ? rustc) [
      pkgs.rustc
      pkgs.cargo
      pkgs.clippy
      pkgs.rustfmt
      pkgs.rust-analyzer
    ]
    ++ lib.optionals (cfg.toolchain.lean.enable && pkgs ? lean4) [ pkgs.lean4 ]
    ++ lib.optionals cfg.toolchain.python.enable [ python-env ]
    ++ lib.optionals (pkgs ? buck2) [ pkgs.buck2 ];
in
{
  inherit
    buck2-toolchain
    packages
    ghc-for-buck2
    ghc-version
    hs-packages-config
    llvm-git
    nvidia-sdk
    python-env
    ;
}
