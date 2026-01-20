# nix/_main.nix
#
# FREESIDEフリーサイド — WHY WAIT?
#
# The directory is the kind signature.
#
{ inputs, lib, ... }:
let
  # Import module indices by kind
  flakeModules = import ./modules/flake/_index.nix { inherit inputs lib; };
  nixosModules = import ./modules/nixos/_index.nix;
  homeModules = import ./modules/home/_index.nix;
in
{
  _class = "flake";

  # Required by flake-parts for perSystem
  systems = import inputs.systems;

  # ════════════════════════════════════════════════════════════════════════════
  # MODULES BY KIND
  #
  # flake.modules.<kind>.<name> for automatic _class validation
  # ════════════════════════════════════════════════════════════════════════════

  flake.modules = {
    flake = {
      inherit (flakeModules)
        build
        build-standalone
        default
        default-with-demos
        devshell
        docs
        formatter
        lint
        nix-conf
        nixpkgs
        std
        nv-sdk
        container
        prelude
        prelude-demos
        options-only
        ;
    };

    nixos = nixosModules;

    home = homeModules;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # OVERLAYS
  #
  # A pure function from the world as it is to the world as it ought to be.
  # ════════════════════════════════════════════════════════════════════════════

  flake.overlays = (import ./overlays inputs).flake.overlays;

  # ════════════════════════════════════════════════════════════════════════════
  # LIB
  #
  # Pure functions. No pkgs, no system.
  # ════════════════════════════════════════════════════════════════════════════

  flake.lib = import ./lib { inherit lib; };

  # ════════════════════════════════════════════════════════════════════════════
  # TEMPLATES
  # ════════════════════════════════════════════════════════════════════════════

  flake.templates = {
    default = {
      path = ./templates/default;
      description = "Standard project";
    };

    nv = {
      path = ./templates/nv;
      description = "NVIDIA/ML project";
    };

    buck2 = {
      path = ./templates/buck2;
      description = "Buck2 build system with hermetic Nix toolchains";
    };

    minimal = {
      path = ./templates/minimal;
      description = "Minimal flake";
    };

    dhall-configured = {
      path = ./templates/dhall-configured;
      description = "Dhall-typed configuration";
    };

    nickel-configured = {
      path = ./templates/nickel-configured;
      description = "Nickel-typed configuration";
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # INTERNAL: aleph-naught's own development
  # ════════════════════════════════════════════════════════════════════════════

  imports = [
    flakeModules.formatter
    flakeModules.lint
    flakeModules.docs
    flakeModules.std
    flakeModules.devshell
    flakeModules.prelude
    flakeModules.prelude-demos
    flakeModules.container
    flakeModules.build
  ];

  perSystem =
    {
      pkgs,
      system,
      ...
    }:
    let
      # WASM infrastructure (internal)
      wasm-infra = import ./prelude/wasm-plugin.nix {
        inherit lib;
        inherit (pkgs) stdenv runCommand;
        inherit (inputs) ghc-wasm-meta;
      };

      # GHC WASM toolchain for compiling .hs packages
      ghc-wasm = inputs.ghc-wasm-meta.packages.${system}.all_9_12 or null;

      # The aleph interface
      # Usage: aleph.eval "Aleph.Packages.Nvidia.nccl" {}
      aleph = import ./prelude/aleph.nix {
        inherit lib pkgs;
        wasmFile = wasm-infra.alephWasm;
      };

      # ────────────────────────────────────────────────────────────────────────
      # // call-package for typed .hs files //
      # ────────────────────────────────────────────────────────────────────────
      # This is the local version for internal use. The prelude exports
      # a version that's available to downstream consumers.
      #
      # User files just need:
      #   module Pkg where
      #   import Aleph.Nix.Package
      #   pkg = mkDerivation [ ... ]
      #
      # The FFI boilerplate is generated automatically.
      #
      call-package =
        path: args:
        let
          pathStr = toString path;
          ext = lib.last (lib.splitString "." pathStr);
          alephModules = ./scripts;

          # Generated Main.hs that wraps the user's package module
          wrapperMain = pkgs.writeText "Main.hs" ''
            {-# LANGUAGE ForeignFunctionInterface #-}
            module Main where

            import Aleph.Nix.Value (Value(..))
            import Aleph.Nix.Derivation (drvToNixAttrs)
            import Aleph.Nix (nixWasmInit)
            import qualified Pkg (pkg)

            main :: IO ()
            main = pure ()

            foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
            initPlugin :: IO ()
            initPlugin = nixWasmInit

            foreign export ccall "pkg" pkgExport :: Value -> IO Value
            pkgExport :: Value -> IO Value
            pkgExport _args = drvToNixAttrs Pkg.pkg
          '';

          # Build single-file Haskell to WASM
          buildHsWasm =
            hsPath:
            let
              name = lib.removeSuffix ".hs" (baseNameOf (toString hsPath));
            in
            pkgs.runCommand "${name}.wasm"
              {
                src = hsPath;
                nativeBuildInputs = [ ghc-wasm ];
              }
              ''
                mkdir -p build
                cd build
                cp -r ${alephModules}/Aleph Aleph
                chmod -R u+w Aleph
                cp $src Pkg.hs
                cp ${wrapperMain} Main.hs
                wasm32-wasi-ghc \
                  -optl-mexec-model=reactor \
                  -optl-Wl,--allow-undefined \
                  -optl-Wl,--export=hs_init \
                  -optl-Wl,--export=nix_wasm_init_v1 \
                  -optl-Wl,--export=pkg \
                  -O2 \
                  Main.hs \
                  -o plugin.wasm
                wasm-opt -O3 plugin.wasm -o $out
              '';
          # All builds go through buildFromSpec (F_ω path only)
          buildSpec = spec: wasm-infra.buildFromSpec { inherit spec pkgs; };
        in
        if ext == "hs" then
          if ghc-wasm == null then
            throw "call-package for .hs files requires ghc-wasm-meta input"
          else if !(builtins ? wasm) then
            throw "call-package for .hs files requires straylight-nix with builtins.wasm"
          else
            let
              wasmDrv = buildHsWasm path;
              spec = builtins.wasm wasmDrv "pkg" args;
            in
            buildSpec spec
        else if ext == "wasm" then
          if !(builtins ? wasm) then
            throw "call-package for .wasm files requires straylight-nix"
          else
            buildSpec (builtins.wasm path "pkg" args)
        else if ext == "nix" then
          pkgs.callPackage path args
        else
          throw "call-package: unsupported extension .${ext}";

      # ────────────────────────────────────────────────────────────────────────
      # // typed packages //
      # ────────────────────────────────────────────────────────────────────────
      # Packages defined in Haskell via call-package using DrvSpec types.
      # Only available when using straylight-nix (builtins.wasm).
      #
      # TODO: Migrate packages to use DrvSpec (Aleph.Nix.DrvSpec) instead of
      # the deleted Derivation.hs types. For now, typed packages are disabled.
      #
      typedPackages = { };

      # aleph-exec: zero-bash build executor (RFC-007)
      aleph-exec = pkgs.callPackage ./packages/aleph-exec.nix { };
    in
    {
      # Make aleph available to other modules via _module.args
      _module.args = {
        inherit aleph call-package aleph-exec;
      };

      packages = {
        mdspan = pkgs.mdspan or null;
        inherit aleph-exec;
      }
      // lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
        llvm-git = pkgs.llvm-git or null;
        nvidia-sdk = pkgs.nvidia-sdk or null;
      }
      // typedPackages;

      checks = import ./checks/default.nix { inherit pkgs system lib; };
    };

  aleph-naught.devshell = {
    enable = true;
    nv.enable = true;
    straylight-nix.enable = true;
  };

  aleph-naught.nixpkgs.nv.enable = true;

  # Buck2 build system integration
  aleph-naught.build = {
    enable = true;
    prelude.enable = true;
    toolchain = {
      cxx.enable = true;
      nv.enable = true;
      haskell.enable = true;
      rust.enable = true;
      lean.enable = true;
      python.enable = true;
    };
  };

  aleph-naught.docs = {
    enable = true;
    title = "Weyl Standard Nix";
    description = "A specification for reproducible, composable infrastructure on Nix";
    theme = "ono-sendai";

    # Document all aleph-naught modules
    modules = [ flakeModules.options-only ];
  };
}
