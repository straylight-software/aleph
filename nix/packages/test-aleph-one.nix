# nix/packages/test-aleph-one.nix
#
# Test the Aleph-1 build system with zlib-ng
#
{
  lib,
  pkgs,
  system,
}:

let
  # Import build infrastructure
  buildLib = import ../build/from-dhall.nix {
    inherit lib;
    inherit (pkgs)
      runCommand
      writeText
      fetchFromGitHub
      fetchurl
      dhall
      ;
    dhall-json = pkgs.dhall-json or null;
  };

  # Import specs
  specLib = import ../build/spec.nix { inherit lib; };

  # Get host triple
  hostTriple = buildLib.systemToTriple system;

  # Build zlib-ng spec
  zlibSpec = specLib.specs.zlib-ng hostTriple;

  # Dep registry - maps dep names to Nix packages
  depRegistry = {
    cmake = pkgs.cmake;
    ninja = pkgs.ninja;
    gnumake = pkgs.gnumake;
    meson = pkgs.meson;
    gcc = pkgs.gcc;
    binutils = pkgs.binutils;

    # For packages with deps
    fmt = pkgs.fmt;
  };

  # aleph-build package
  aleph-build = pkgs.callPackage ./aleph-build.nix { };

  # Builders map
  builders = {
    cmake = aleph-build;
    autotools = aleph-build;
    meson = aleph-build;
    headerOnly = aleph-build;
  };

in
buildLib.buildFromDhall {
  spec = zlibSpec;
  inherit depRegistry builders system;
  cores = 8;
}
