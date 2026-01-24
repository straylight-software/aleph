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
        buck2
        build-standalone
        default
        default-with-demos
        devshell
        docs
        formatter
        full
        lint
        lre
        nativelink
        nix-conf
        nixpkgs
        nv-sdk
        container
        prelude
        prelude-demos
        shortlist
        shortlist-standalone
        std
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

  flake.lib = import ./lib { inherit lib; } // {
    # Buck2 builder - use from downstream flakes:
    #   packages.myapp = aleph.lib.buck2.build pkgs { target = "//src:myapp"; };
    buck2 = import ./lib/buck2.nix { inherit inputs lib; };
  };

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
    flakeModules.buck2
    flakeModules.shortlist
    flakeModules.lre
    # nix2gpu.flakeModule must be imported before nativelink module
    # (provides perSystem.nix2gpu options)
    inputs.nix2gpu.flakeModule
    flakeModules.nativelink
  ];

  # Enable shortlist, LRE, and NativeLink containers for aleph itself
  aleph-naught.shortlist.enable = true;
  aleph-naught.lre.enable = true;
  aleph-naught.nativelink.enable = true;

  perSystem =
    {
      pkgs,
      system,
      config,
      ...
    }:
    let
      # WASM infrastructure (internal)
      wasm-infra = import ./prelude/wasm-plugin.nix {
        inherit lib;
        inherit (pkgs) stdenv runCommand writeText;
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
      # IFD AVOIDANCE: If a pre-built .wasm file exists alongside the .hs file,
      # we use it directly instead of building at evaluation time. This avoids
      # the "import from derivation" warning in `nix flake show`.
      #
      # To pre-build all .wasm files:
      #   nix build .#wasm-packages -o result-wasm
      #   cp result-wasm/*.wasm nix/packages/
      #
      call-package =
        path: args:
        let
          pathStr = toString path;
          ext = lib.last (lib.splitString "." pathStr);
          alephModules = ../src/tools/scripts;

          # Check for pre-built WASM file (avoids IFD)
          baseName = lib.removeSuffix ".hs" (baseNameOf pathStr);
          prebuiltWasm = ./packages + "/${baseName}.wasm";
          hasPrebuiltWasm = builtins.pathExists prebuiltWasm;

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
          if !(builtins ? wasm) then
            throw "call-package for .hs files requires straylight-nix with builtins.wasm"
          # Use pre-built WASM if available (no IFD)
          else if hasPrebuiltWasm then
            wasm-infra.buildFromSpec {
              spec = builtins.wasm prebuiltWasm "pkg" args;
              inherit pkgs;
            }
          # Fall back to building at eval time (causes IFD warning)
          else if ghc-wasm == null then
            throw "call-package for .hs files requires ghc-wasm-meta input (or pre-built ${baseName}.wasm)"
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
      # List of all .hs package files (used for both building and pre-building WASM)
      hsPackageFiles = [
        "test-hello"
        "test-zlib-ng"
        "test-tool-deps"
        "test-typed-tools"
        "catch2"
        "fmt"
        "mdspan"
        "nlohmann-json"
        "rapidjson"
        "spdlog"
        "zlib-ng"
        "nvidia-nccl"
        "nvidia-cudnn"
        "nvidia-tensorrt"
        "nvidia-cutensor"
        "nvidia-cusparselt"
        "nvidia-cutlass"
      ];

      # Build WASM from a single .hs file (for pre-building)
      buildHsWasmStandalone =
        name:
        let
          hsPath = ./packages + "/${name}.hs";
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
        in
        pkgs.runCommand "${name}.wasm"
          {
            src = hsPath;
            nativeBuildInputs = [ ghc-wasm ];
          }
          ''
            mkdir -p build
            cd build
            cp -r ${../src/tools/scripts}/Aleph Aleph
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

      # All WASM files bundled together (for easy copying to repo)
      wasmPackagesBundle = lib.optionalAttrs (ghc-wasm != null) {
        wasm-packages = pkgs.runCommand "wasm-packages" { } ''
          mkdir -p $out
          ${lib.concatMapStringsSep "\n" (name: ''
            cp ${buildHsWasmStandalone name} $out/${name}.wasm
          '') hsPackageFiles}
        '';
      };

      typedPackages = lib.optionalAttrs (builtins ? wasm) (
        lib.listToAttrs (
          map (name: {
            inherit name;
            value = call-package (./packages + "/${name}.hs") { };
          }) hsPackageFiles
        )
      );

      # NativeLink from inputs (for LRE)
      nativelink =
        if inputs ? nativelink then
          inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink
            or null
        else
          null;
    in
    {
      # Make aleph available to other modules via _module.args
      _module.args = {
        inherit aleph call-package;
      };

      # Wire up shortlist paths to buck2 config
      buck2.shortlist = {
        fmt = "${pkgs.fmt}";
        fmt_dev = "${pkgs.fmt.dev}";
        zlib_ng = "${pkgs.zlib-ng}";
        catch2 = "${pkgs.catch2_3}";
        catch2_dev = "${pkgs.catch2_3.dev or pkgs.catch2_3}";
        spdlog = "${pkgs.spdlog}";
        spdlog_dev = "${pkgs.spdlog.dev or pkgs.spdlog}";
        mdspan = "${pkgs.mdspan}";
        rapidjson = "${pkgs.rapidjson}";
        nlohmann_json = "${pkgs.nlohmann_json}";
        libsodium = "${pkgs.libsodium}";
        libsodium_dev = "${pkgs.libsodium.dev or pkgs.libsodium}";
      };

      packages = {
        mdspan = pkgs.mdspan or null;
        wsn-lint = pkgs.callPackage ./packages/wsn-lint.nix { };

        # Buck2 built packages - these can be used in NixOS, containers, etc.
        # fmt-test = config.buck2.build { target = "//examples/cxx:fmt_test"; };
      }
      // lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
        llvm-git = pkgs.llvm-git or null;
        nvidia-sdk = pkgs.nvidia-sdk or null;
      }
      // lib.optionalAttrs (nativelink != null) {
        inherit nativelink;
      }
      // wasmPackagesBundle
      // typedPackages;

      checks = import ./checks/default.nix { inherit pkgs system lib; };

      # nix2gpu requires explicit empty default (upstream bug - no default in option)
      nix2gpu = { };
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
      haskell = {
        enable = true;
        # Core packages for Buck2 haskell_binary rules
        # Includes Aleph.Script dependencies for typed CLI tools
        packages =
          hp:
          builtins.filter (p: p != null) [
            # Core
            hp.text or null
            hp.bytestring or null
            hp.containers or null
            hp.directory or null
            hp.filepath or null
            hp.process or null
            hp.time or null

            # CLI / scripting
            hp.aeson or null
            hp.aeson-pretty or null
            hp.optparse-applicative or null
            hp.megaparsec or null
            hp.prettyprinter or null

            # Aleph.Script dependencies
            hp.shelly or null
            hp.foldl or null
            hp.dhall or null
            hp.crypton or null
            hp.memory or null
            hp.unordered-containers or null
            hp.vector or null
            hp.unix or null
            hp.async or null
            hp.transformers or null
            hp.mtl or null
          ];
      };
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
