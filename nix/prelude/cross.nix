# nix/prelude/cross.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // cross //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Wintermute,' Case said, 'you told me you were just a part of
#      something else. Like a lobe of a brain.' He kept his eyes on
#      the back of 3Jane's head. 'You said you were trying to get
#      to where you could put yourself together.'
#
#     'Yes.'
#
#     'And Armitage? Corto? Where's he fit in?'
#
#     'Corto, Case. What you think of as Armitage was just something
#      I built from the wreckage of Corto. It was necessary, Case.
#      Necessary as your own involvement.'
#
#                                                         — Neuromancer
#
# Cross-compilation targets. Building for architectures other than the
# one you're sitting on. Grace, Jetson, aarch64, x86_64 — the targets
# we build toward.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  final,
  platform,
  gpu,
  turing-registry,
}:

let
  # lisp-case aliases for lib functions
  inherit (lib) optionalAttrs;
  # :: t13
  optional-attrs = optionalAttrs;

  # lisp-case aliases for nixpkgs cross-compilation
  # :: t14
  inherit (final) pkgsCross;
  pkgs-cross = pkgsCross;

  # :: t15
  # :: t16
  # :: t17
  # lisp-case aliases for gpu targets
  inherit (gpu) sm_90 sm_90a sm_120;
  sm-90 = sm_90;
  sm-90a = sm_90a;
  sm-120 = sm_120;

  # lisp-case wrapper for cross-compilation stdenv
  # wraps pkgs.stdenv.mkDerivation with turing registry and target metadata
  mk-cross-derivation =
    target-name: cross-pkgs: args:
    let
      inherit (cross-pkgs) stdenv;
      inherit (stdenv) mkDerivation;
    in
    mkDerivation (
      args
      // turing-registry.attrs
      // {
        NIX_CFLAGS_COMPILE = (args.NIX_CFLAGS_COMPILE or "") + " " + turing-registry.cxxflags-str;
        passthru = (args.passthru or { }) // {
          aleph-target = target-name;
        };
      }
    );

  x86-targets = {
    # Grace Hopper: aarch64 + Hopper GPU
    grace = rec {
      name = "grace";
      arch = "aarch64";
      target-gpu = sm-90a;
      pkgs = pkgs-cross.aarch64-multiplatform;

      mk-derivation = mk-cross-derivation name pkgs;

      __functor = _self: mk-derivation;
    };

    # Jetson Thor: aarch64 + Thor GPU
    jetson = rec {
      name = "jetson";
      arch = "aarch64";
      target-gpu = sm-90;
      pkgs = pkgs-cross.aarch64-multiplatform;

      mk-derivation = mk-cross-derivation name pkgs;

      __functor = _self: mk-derivation;
    };

    # Generic aarch64: no GPU
    aarch64 = rec {
      name = "aarch64";
      arch = "aarch64";
      target-gpu = gpu.none;
      pkgs = pkgs-cross.aarch64-multiplatform;

      mk-derivation = mk-cross-derivation name pkgs;

      __functor = _self: mk-derivation;
    };
  };

  arm-targets = {
    # Reverse: aarch64 → x86_64
    x86-64 = rec {
      name = "x86-64";
      arch = "x86_64";
      target-gpu = sm-120;
      pkgs = pkgs-cross.gnu64;

      mk-derivation = mk-cross-derivation name pkgs;

      __functor = _self: mk-derivation;
    };
  };
in

# ────────────────────────────────────────────────────────────────────────────
#                          // targets //
# ────────────────────────────────────────────────────────────────────────────

if platform.is-x86 then
  x86-targets
else if platform.is-arm then
  arm-targets
else
  { }
