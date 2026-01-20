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

  devshell = import ./devshell.nix { };
  docs = import ./docs.nix { inherit inputs; };
  formatter = import ./formatter.nix { inherit inputs; };
  lint = import ./lint.nix { };
  nix-conf = import ./nix-conf.nix { };
  nixpkgs = import ./nixpkgs.nix { inherit inputs; };
  std = import ./std.nix { inherit inputs; };
  nv-sdk = import ./nv-sdk.nix;
  container = import ./container { inherit lib; };
  prelude = import ./prelude.nix { inherit inputs; };
  prelude-demos = import ./prelude-demos.nix;

  # Options-only module for documentation generation
  options-only =
    { lib, ... }:
    let
      schema = import ./options-schema.nix { inherit lib; };
    in
    {
      options.aleph-naught = schema;
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

in
{
  inherit
    container
    default
    default-with-demos
    devshell
    docs
    formatter
    lint
    nix-conf
    nixpkgs
    nv-sdk
    options-only
    prelude
    prelude-demos
    std
    ;
}
