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
            wasm-infra.buildFromSpec { inherit spec pkgs; }
        else if ext == "wasm" then
          if !(builtins ? wasm) then
            throw "call-package for .wasm files requires straylight-nix"
          else
            wasm-infra.buildFromSpec {
              spec = builtins.wasm path "pkg" args;
              inherit pkgs;
            }
        else if ext == "nix" then
          pkgs.callPackage path args
        else
          throw "call-package: unsupported extension .${ext}";

      # ────────────────────────────────────────────────────────────────────────
      # // typed packages //
      # ────────────────────────────────────────────────────────────────────────
      # Packages defined in Haskell via call-package.
      # Only available when using straylight-nix (builtins.wasm).
      #
      typedPackages = lib.optionalAttrs (builtins ? wasm && ghc-wasm != null) {
        # Test packages
        test-hello = call-package ./packages/test-hello.hs { };
        test-zlib-ng = call-package ./packages/test-zlib-ng.hs { };
        test-tool-deps = call-package ./packages/test-tool-deps.hs { };
        test-typed-tools = call-package ./packages/test-typed-tools.hs { };

        # C++ libraries
        catch2 = call-package ./packages/catch2.hs { };
        fmt = call-package ./packages/fmt.hs { };
        mdspan = call-package ./packages/mdspan.hs { };
        nlohmann-json = call-package ./packages/nlohmann-json.hs { };
        rapidjson = call-package ./packages/rapidjson.hs { };
        spdlog = call-package ./packages/spdlog.hs { };
        zlib-ng = call-package ./packages/zlib-ng.hs { };
        # Note: abseil-cpp uses libmodern overlay (needs combine-archive)

        # NVIDIA SDK (from PyPI wheels)
        nvidia-nccl = call-package ./packages/nvidia-nccl.hs { };
        nvidia-cudnn = call-package ./packages/nvidia-cudnn.hs { };
        nvidia-tensorrt = call-package ./packages/nvidia-tensorrt.hs { };
        nvidia-cutensor = call-package ./packages/nvidia-cutensor.hs { };
        nvidia-cusparselt = call-package ./packages/nvidia-cusparselt.hs { };
        nvidia-cutlass = call-package ./packages/nvidia-cutlass.hs { };
      };
    in
    {
      # Make aleph available to other modules via _module.args
      _module.args = {
        inherit aleph call-package;
      };

      packages = {
        mdspan = pkgs.mdspan or null;
        wsn-lint = pkgs.callPackage ./packages/wsn-lint.nix { };
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
