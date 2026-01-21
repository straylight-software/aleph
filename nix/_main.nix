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
      # aleph-exec: zero-bash build executor (RFC-007)
      # Defined first so it's available for buildFromSpec
      aleph-exec = pkgs.callPackage ./packages/aleph-exec.nix { };

      # WASM infrastructure (internal)
      wasm-infra = import ./prelude/wasm-plugin.nix {
        inherit lib aleph-exec;
        inherit (pkgs) stdenv runCommand;
        inherit (inputs) ghc-wasm-meta;
      };

      # GHC WASM toolchain for compiling .hs packages
      ghc-wasm = inputs.ghc-wasm-meta.packages.${system}.all_9_12 or null;

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
            import Aleph.Nix.DrvSpec (drvToNix)
            import Aleph.Nix (nixWasmInit)
            import qualified Pkg (pkg)

            main :: IO ()
            main = pure ()

            foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
            initPlugin :: IO ()
            initPlugin = nixWasmInit

            foreign export ccall "pkg" pkgExport :: Value -> IO Value
            pkgExport :: Value -> IO Value
            pkgExport _args = drvToNix Pkg.pkg
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
          # wasmDrv is passed to include it in derivation hash for cache invalidation
          buildSpec = wasmDrv: spec: wasm-infra.buildFromSpec { inherit spec pkgs wasmDrv; };
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
            buildSpec wasmDrv spec
        else if ext == "wasm" then
          if !(builtins ? wasm) then
            throw "call-package for .wasm files requires straylight-nix"
          else
            buildSpec path (builtins.wasm path "pkg" args)
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
      # F_ω Architecture (RFC-007): All packages use Aleph.Nix.DrvSpec.
      # Dhall is the substrate - Haskell emits Dhall, Nix resolves paths,
      # aleph-exec executes typed actions.
      #
      typedPackages = lib.optionalAttrs (builtins ? wasm && ghc-wasm != null) {
        # C++ Libraries
        zlib-ng = call-package ./packages/zlib-ng.hs { };
        fmt = call-package ./packages/fmt.hs { };
        mdspan = call-package ./packages/mdspan.hs { };
        rapidjson = call-package ./packages/rapidjson.hs { };
        nlohmann-json = call-package ./packages/nlohmann-json.hs { };
        catch2 = call-package ./packages/catch2.hs { };
        spdlog = call-package ./packages/spdlog.hs { };
        libsodium = call-package ./packages/libsodium.hs { };

        # NVIDIA SDK components
        nvidia-cudnn = call-package ./packages/nvidia-cudnn.hs { };
        nvidia-cusparselt = call-package ./packages/nvidia-cusparselt.hs { };
        nvidia-cutensor = call-package ./packages/nvidia-cutensor.hs { };
        nvidia-cutlass = call-package ./packages/nvidia-cutlass.hs { };
        nvidia-nccl = call-package ./packages/nvidia-nccl.hs { };
        nvidia-tensorrt = call-package ./packages/nvidia-tensorrt.hs { };

        # Test packages
        test-hello = call-package ./packages/test-hello.hs { };
        test-tool-deps = call-package ./packages/test-tool-deps.hs { };
        test-typed-tools = call-package ./packages/test-typed-tools.hs { };
        test-zero-bash = call-package ./packages/test-zero-bash.hs { };
        test-zlib-ng = call-package ./packages/test-zlib-ng.hs { };
      };
    in
    {
      # Make aleph available to other modules via _module.args
      _module.args = {
        inherit call-package aleph-exec;
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
