# nix/prelude/types/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // typed prelude //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Dhall types compiled to Nix at build time. The Dhall is the spec,
# this is the runtime. Always in sync.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ pkgs }:
let
  dhall-src = ./.; # This directory contains the .dhall files

  # Generate Nix from Dhall at build time
  generated =
    pkgs.runCommand "prelude-types"
      {
        nativeBuildInputs = [ pkgs.haskellPackages.dhall-nix ];
      }
      ''
        mkdir -p $out
        cd ${dhall-src}
        dhall-to-nix <<< './Target.dhall' > $out/target.nix
        dhall-to-nix <<< './Toolchain.dhall' > $out/toolchain.nix
      '';

  target = import (generated + "/target.nix");
  toolchain = import (generated + "/toolchain.nix");

in
{
  inherit target toolchain;

  # Re-export for convenience
  inherit (target)
    Arch
    OS
    ABI
    Cpu
    Gpu
    Triple
    x86-64-linux
    x86-64-linux-musl
    grace
    orin
    thor
    aarch64-darwin
    wasm32-wasi
    to-string
    ;

  inherit (toolchain)
    Hash
    Artifact
    OptLevel
    LTOMode
    DebugInfo
    Flag
    CompilerKind
    Compiler
    Linker
    CxxStdlib
    Toolchain
    nv-clang
    clang
    gcc
    rustc
    ghc
    lean
    native-toolchain
    cross-toolchain
    ;
}
