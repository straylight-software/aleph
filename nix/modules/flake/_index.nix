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

  # :: t25
  # :: t27
  # :: t29
  # :: t31
  # :: t33
  # :: t35
  # :: t37
  # :: t39
  # :: t41
  # :: t43
  # :: t45
  # :: t47
  # :: Any
  # :: t50
  # :: t52
  # :: Any
  build = import ./build/flake-module.nix { inherit inputs; };
  buck2 = import ./buck2.nix { inherit inputs; };
  # :: { lib : t54 } | ... -> {}
  devshell = import ./devshell.nix { };
  docs = import ./docs.nix { inherit inputs; };
  # :: t57
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
  # :: { _class : "flake", imports : [Any] }
  # :: "flake"
  prelude = import ./prelude.nix { inherit inputs; };
  # :: [Any]
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

  # :: { _class : "flake", imports : [Any] }
  # :: "flake"
  # ──────────────────────────────────────────────────────────────────────────
  # :: [Any]
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
  # :: { _class : "flake", imports : [t43] }
  # :: "flake"
  };
# :: [t43]

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
      # :: { _class : "flake", imports : [t45] }
      # :: "flake"
      prelude-demos
      # :: [t45]
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

    # :: { _class : "flake", imports : [Any] }
    # :: "flake"
    imports = [
      # :: [Any]
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
