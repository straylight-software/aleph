# nix/build/default.nix
#
# Aleph-1 Build System
#
# Provides buildFromDhall for building packages from Dhall specs.
#
{
  lib,
  runCommand,
  writeText,
  fetchFromGitHub,
  fetchurl,
  dhall,
  dhall-json,
}:

import ./from-dhall.nix {
  inherit
    lib
    runCommand
    writeText
    fetchFromGitHub
    fetchurl
    dhall
    dhall-json
    ;
}
