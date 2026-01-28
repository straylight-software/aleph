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
  optional-attrs = lib.optionalAttrs;

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
      # NativeLink from inputs (for LRE)
      nativelink = inputs.nativelink.packages.${system}.default;
    in
    {

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

        # Armitage - daemon-free Nix operations via Buck2
        armitage = inputs.self.lib.buck2.build pkgs {
          src = inputs.self;
          target = "//src/armitage:armitage";
        };
        armitage-proxy = inputs.self.lib.buck2.build pkgs {
          src = inputs.self;
          target = "//src/armitage:armitage-proxy";
        };
      }
      // optional-attrs (pkgs ? mdspan) { inherit (pkgs) mdspan; }
      // optional-attrs (system == "x86_64-linux" || system == "aarch64-linux") (
        optional-attrs (pkgs ? llvm-git) { inherit (pkgs) llvm-git; }
        // optional-attrs (pkgs ? nvidia-sdk) { inherit (pkgs) nvidia-sdk; }
      )
      // optional-attrs (nativelink != null) {
        inherit nativelink;
      };

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
    remote.enable = true; # Fly.io remote execution
    toolchain = {
      cxx.enable = true;
      nv.enable = true;
      haskell.enable = true;
      # Package list is in nix/modules/flake/build/options.nix default
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
