# nix/_main.nix
#
# FREESIDEフリーサイド — WHY WAIT?
#
# The directory is the kind signature.
#
{ inputs, lib, ... }:
let
  # ════════════════════════════════════════════════════════════════════════════
  # LISP-CASE ALIASES
  #
  # Local aliases for lib.* and builtins.* functions to satisfy ALEPH-E003.
  # External API names (nixpkgs attributes, flake outputs) remain unchanged.
  # ════════════════════════════════════════════════════════════════════════════
  split-string = lib.splitString;
  remove-suffix = lib.removeSuffix;
  optional-attrs = lib.optionalAttrs;
  concat-map-strings-sep = lib.concatMapStringsSep;
  list-to-attrs = lib.listToAttrs;
  path-exists = builtins.pathExists;
  base-name-of = builtins.baseNameOf;
  to-string = builtins.toString;

  # Import module indices by kind
  flake-modules = import ./modules/flake/_index.nix { inherit inputs lib; };
  nixos-modules = import ./modules/nixos/_index.nix;
  home-modules = import ./modules/home/_index.nix;
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
      inherit (flake-modules)
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

    nixos = nixos-modules;

    home = home-modules;
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
    buck2 = import ./lib/buck2.nix { inherit inputs; };
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
  # INTERNAL: aleph's own development
  # ════════════════════════════════════════════════════════════════════════════

  imports = [
    flake-modules.formatter
    flake-modules.lint
    flake-modules.docs
    flake-modules.std
    flake-modules.devshell
    flake-modules.prelude
    flake-modules.prelude-demos
    flake-modules.container
    flake-modules.build
    flake-modules.buck2
    flake-modules.shortlist
    flake-modules.lre
    # nix2gpu.flakeModule must be imported before nativelink module
    # (provides perSystem.nix2gpu options)
    inputs.nix2gpu.flakeModule
    flake-modules.nativelink
  ];

  # Enable shortlist, LRE, and NativeLink containers for aleph itself
  aleph.shortlist.enable = true;
  aleph.lre.enable = true;
  aleph.nativelink.enable = true;

  "perSystem" =
    {
      pkgs,
      system,
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
      ghc-wasm = inputs.ghc-wasm-meta.packages.${system}.all_9_12;

      # The aleph interface
      # Usage: aleph.eval "Aleph.Packages.Nvidia.nccl" {}
      aleph = import ./prelude/aleph.nix {
        inherit lib pkgs;
        "wasmFile" = wasm-infra.alephWasm;
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
          path-str = to-string path;
          ext = lib.last (split-string "." path-str);
          aleph-modules = ../src/tools/scripts;

          # Check for pre-built WASM file (avoids IFD)
          base-name = remove-suffix ".hs" (base-name-of path-str);
          prebuilt-wasm = ./packages + "/${base-name}.wasm";
          has-prebuilt-wasm = path-exists prebuilt-wasm;

          # Generated Main.hs that wraps the user's package module
          wrapper-main = ./build/templates/wasm-main.hs;

          # Build single-file Haskell to WASM
          build-hs-wasm =
            hs-path:
            let
              name = remove-suffix ".hs" (base-name-of (to-string hs-path));
            in
            pkgs.runCommand "${name}.wasm"
              {
                src = hs-path;
                "nativeBuildInputs" = [ ghc-wasm ];
              }
              ''
                mkdir -p build && cd build
                cp -r ${aleph-modules}/Aleph Aleph
                chmod -R u+w Aleph
                cp $src Pkg.hs
                cp ${wrapper-main} Main.hs
                wasm32-wasi-ghc \
                  -optl-mexec-model=reactor \
                  -optl-Wl,--allow-undefined \
                  -optl-Wl,--export=hs_init \
                  -optl-Wl,--export=nix_wasm_init_v1 \
                  -optl-Wl,--export=pkg \
                  -O2 Main.hs -o plugin.wasm
                wasm-opt -O3 plugin.wasm -o $out
              '';
        in
        if ext == "hs" then
          if !(builtins ? wasm) then
            throw "call-package for .hs files requires straylight-nix with builtins.wasm"
          # Use pre-built WASM if available (no IFD)
          else if has-prebuilt-wasm then
            wasm-infra.build-from-spec {
              spec = builtins.wasm prebuilt-wasm "pkg" args;
              inherit pkgs;
            }
          # Fall back to building at eval time (causes IFD warning)
          else if ghc-wasm == null then
            throw "call-package for .hs files requires ghc-wasm-meta input (or pre-built ${base-name}.wasm)"
          else
            let
              wasm-drv = build-hs-wasm path;
              spec = builtins.wasm wasm-drv "pkg" args;
            in
            wasm-infra.build-from-spec { inherit spec pkgs; }
        else if ext == "wasm" then
          if !(builtins ? wasm) then
            throw "call-package for .wasm files requires straylight-nix"
          else
            wasm-infra.build-from-spec {
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
      hs-package-files = [
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
      build-hs-wasm-standalone =
        name:
        let
          hs-path = ./packages + "/${name}.hs";
          wrapper-main = ./build/templates/wasm-main.hs;
          aleph-modules = ../src/tools/scripts;
        in
        pkgs.runCommand "${name}.wasm"
          {
            src = hs-path;
            "nativeBuildInputs" = [ ghc-wasm ];
          }
          ''
            mkdir -p build && cd build
            cp -r ${aleph-modules}/Aleph Aleph
            chmod -R u+w Aleph
            cp $src Pkg.hs
            cp ${wrapper-main} Main.hs
            wasm32-wasi-ghc \
              -optl-mexec-model=reactor \
              -optl-Wl,--allow-undefined \
              -optl-Wl,--export=hs_init \
              -optl-Wl,--export=nix_wasm_init_v1 \
              -optl-Wl,--export=pkg \
              -O2 Main.hs -o plugin.wasm
            wasm-opt -O3 plugin.wasm -o $out
          '';

      # All WASM files bundled together (for easy copying to repo)
      wasm-packages-bundle = optional-attrs (ghc-wasm != null) {
        wasm-packages = pkgs.runCommand "wasm-packages" { } ''
          mkdir -p $out
          ${concat-map-strings-sep "\n" (name: ''
            cp ${build-hs-wasm-standalone name} $out/${name}.wasm
          '') hs-package-files}
        '';
      };

      typed-packages = optional-attrs (builtins ? wasm) (
        list-to-attrs (
          map (name: {
            inherit name;
            value = call-package (./packages + "/${name}.hs") { };
          }) hs-package-files
        )
      );

      # NativeLink from inputs (for LRE)
      nativelink = inputs.nativelink.packages.${system}.default;
    in
    {
      # Make aleph available to other modules via _module.args
      _module.args = {
        inherit aleph call-package;
      };

      # Wire up shortlist paths to buck2 config
      buck2.shortlist = {
        fmt = "${pkgs.fmt}";
        "fmt_dev" = "${pkgs.fmt.dev}";
        "zlib_ng" = "${pkgs.zlib-ng}";
        catch2 = "${pkgs.catch2_3}";
        "catch2_dev" = "${pkgs.catch2_3.dev or pkgs.catch2_3}";
        spdlog = "${pkgs.spdlog}";
        "spdlog_dev" = "${pkgs.spdlog.dev or pkgs.spdlog}";
        mdspan = "${pkgs.mdspan}";
        rapidjson = "${pkgs.rapidjson}";
        "nlohmann_json" = "${pkgs.nlohmann_json}";
        libsodium = "${pkgs.libsodium}";
        "libsodium_dev" = "${pkgs.libsodium.dev or pkgs.libsodium}";
      };

      packages = {
        aleph-lint = pkgs.callPackage ./packages/aleph-lint.nix { };

        # Armitage - daemon-free Nix operations
        # Built via GHC from overlay, not Buck2 (faster for Nix builds)
        # For Buck2: buck2 build //src/armitage:armitage
        armitage = pkgs.armitage-cli;
        inherit (pkgs) armitage-proxy;
      }
      // optional-attrs (pkgs ? mdspan) { inherit (pkgs) mdspan; }
      // optional-attrs (system == "x86_64-linux" || system == "aarch64-linux") (
        optional-attrs (pkgs ? llvm-git) { inherit (pkgs) llvm-git; }
        // optional-attrs (pkgs ? nvidia-sdk) { inherit (pkgs) nvidia-sdk; }
      )
      // optional-attrs (nativelink != null) {
        inherit nativelink;
      }
      // wasm-packages-bundle
      // typed-packages;

      checks = import ./checks/default.nix { inherit pkgs system lib; };

      # nix2gpu requires explicit empty default (upstream bug - no default in option)
      nix2gpu = { };
    };

  aleph.devshell = {
    enable = true;
    nv.enable = true;
    straylight-nix.enable = true;
  };

  aleph.nixpkgs.nv.enable = true;

  # Buck2 build system integration
  aleph.build = {
    enable = true;
    prelude.enable = true;
    toolchain = {
      cxx.enable = true;
      nv.enable = true;
      haskell = {
        enable = true;
        # Extra packages for Buck2 haskell_binary rules
        # Note: Core packages (text, bytestring, containers, time, etc.) ship with GHC 9.8+
        packages = hp: [
          hp.aeson
          hp.aeson-pretty
          hp.optparse-applicative
          hp.megaparsec
          hp.prettyprinter
          hp.shelly
          hp.foldl
          hp.dhall
          hp.crypton
          hp.memory
          hp.unordered-containers
          hp.vector
          hp.async

          # Armitage proxy (TLS MITM, certificate generation)
          hp.network
          hp.tls # version pinned in haskell.nix overlay
          hp.crypton-x509
          hp.crypton-x509-store
          hp.data-default-class
          hp.pem
          hp.asn1-types
          hp.asn1-encoding
          hp.hourglass

          # gRPC for NativeLink CAS integration
          # proto-lens-setup patched for Cabal 3.14+ in haskell.nix
          hp.grapesy
        ];
      };
      rust.enable = true;
      lean.enable = true;
      python.enable = true;
    };
  };

  aleph.docs = {
    enable = true;
    title = "Weyl Standard Nix";
    description = "A specification for reproducible, composable infrastructure on Nix";
    theme = "ono-sendai";

    # Document all aleph modules
    modules = [ flake-modules.options-only ];
  };
}
