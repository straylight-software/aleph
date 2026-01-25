# nix/modules/flake/_index.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                                             // flake modules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games, in early
#     graphics programs and military experimentation with cranial
#     jacks. On the Sony, a two-dimensional space war faded behind
#     a forest of mathematically generated ferns, demonstrating the
#     spatial possibilities of logarithmic spirals.
#
#                                                         — Neuromancer
#
# Index of all flake modules (:: FlakeModule). The directory is the
# kind signature.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs, lib }:
let
  # ──────────────────────────────────────────────────────────────────────────
  #                                                      // individual modules
  # ──────────────────────────────────────────────────────────────────────────

  build = import ./build/flake-module.nix { inherit inputs; };
  buck2 = import ./buck2.nix { inherit inputs; };
  devshell = import ./devshell.nix { };
  docs = import ./docs.nix { inherit inputs; };
  formatter = import ./formatter.nix { inherit inputs; };
  lint = import ./lint.nix { };
  lre = import ./lre.nix { inherit inputs; };
  nativelink = import ./nativelink/flake-module.nix { inherit inputs; };
  nix-conf = import ./nix-conf.nix { };
  nixpkgs = import ./nixpkgs.nix { inherit inputs; };
  shortlist = import ./shortlist.nix { inherit inputs; };
  std = import ./std.nix { inherit inputs; };
  nv-sdk = import ./nv-sdk.nix;
  container = import ./container { inherit inputs lib; };
  prelude = import ./prelude.nix { inherit inputs; };
  prelude-demos = import ./prelude-demos.nix;

  # Options-only module for documentation generation
  options-only =
    { lib, ... }:
    let
      schema = import ./options-schema.nix { inherit lib; };
    in
    {
      options.aleph = schema;
    };

  # ──────────────────────────────────────────────────────────────────────────
  #                                                              // composites
  # ──────────────────────────────────────────────────────────────────────────

  # // batteries // included //
  default = {
    _class = "flake";

    imports = [
      formatter
      lint
      docs
      std
      devshell
      prelude
      nv-sdk
      container
    ];
  };

  # // demo // test //
  default-with-demos = {
    _class = "flake";

    imports = [
      formatter
      lint
      docs
      std
      devshell
      prelude
      prelude-demos
      nv-sdk
      container
    ];
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                                                    // build module export
  # ──────────────────────────────────────────────────────────────────────────
  # Standalone build module for downstream flakes that just want Buck2
  # without the full aleph devshell
  build-standalone = {
    _class = "flake";

    imports = [
      build
      nixpkgs # Required for overlays
    ];
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                                                // shortlist module export
  # ──────────────────────────────────────────────────────────────────────────
  # Standalone shortlist module: hermetic C++ libraries + Buck2 toolchain
  # Usage:
  #   imports = [ aleph.modules.flake.shortlist-standalone ];
  #   aleph.shortlist.enable = true;
  shortlist-standalone = {
    _class = "flake";

    imports = [
      build
      shortlist
      nixpkgs # Required for overlays
    ];
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                                                    // full stack export
  # ──────────────────────────────────────────────────────────────────────────
  # Complete aleph build infrastructure for downstream flakes:
  #   - LLVM 22 hermetic toolchain (Buck2 integration)
  #   - Shortlist C++ libraries (fmt, spdlog, etc.)
  #   - NativeLink Local Remote Execution
  #
  # Usage in downstream flake.nix:
  #
  #   inputs.aleph.url = "github:straylight-software/aleph";
  #
  #   outputs = { self, aleph, ... }:
  #     aleph.inputs.flake-parts.lib.mkFlake { inherit inputs; } {
  #       imports = [ aleph.modules.flake.full ];
  #
  #       aleph = {
  #         build.enable = true;
  #         shortlist.enable = true;
  #         lre.enable = true;
  #       };
  #     };
  #
  full = {
    _class = "flake";

    imports = [
      build
      shortlist
      lre
      devshell
      nixpkgs
    ];
  };

in
{
  inherit
    build
    buck2
    build-standalone
    container
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
    options-only
    prelude
    prelude-demos
    shortlist
    shortlist-standalone
    std
    ;
}
