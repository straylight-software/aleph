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

# ────────────────────────────────────────────────────────────────────────────
#                          // x86 -> arm targets //
# ────────────────────────────────────────────────────────────────────────────

lib.optionalAttrs platform.is-x86 {
  # Grace Hopper: aarch64 + Hopper GPU
  grace = rec {
    name = "grace";
    arch = "aarch64";
    target-gpu = gpu.sm_90a;
    pkgs = final.pkgsCross.aarch64-multiplatform;

    mkDerivation =
      args:
      pkgs.stdenv.mkDerivation (
        args
        // turing-registry.attrs
        // {
          NIX_CFLAGS_COMPILE = (args.NIX_CFLAGS_COMPILE or "") + " " + turing-registry.cxxflags-str;
          passthru = (args.passthru or { }) // {
            straylight-target = name;
          };
        }
      );

    __functor = _self: mkDerivation;
  };

  # Jetson Thor: aarch64 + Thor GPU
  jetson = rec {
    name = "jetson";
    arch = "aarch64";
    target-gpu = gpu.sm_90;
    pkgs = final.pkgsCross.aarch64-multiplatform;

    mkDerivation =
      args:
      pkgs.stdenv.mkDerivation (
        args
        // turing-registry.attrs
        // {
          NIX_CFLAGS_COMPILE = (args.NIX_CFLAGS_COMPILE or "") + " " + turing-registry.cxxflags-str;
          passthru = (args.passthru or { }) // {
            straylight-target = name;
          };
        }
      );

    __functor = _self: mkDerivation;
  };

  # Generic aarch64: no GPU
  aarch64 = rec {
    name = "aarch64";
    arch = "aarch64";
    target-gpu = gpu.none;
    pkgs = final.pkgsCross.aarch64-multiplatform;

    mkDerivation =
      args:
      pkgs.stdenv.mkDerivation (
        args
        // turing-registry.attrs
        // {
          NIX_CFLAGS_COMPILE = (args.NIX_CFLAGS_COMPILE or "") + " " + turing-registry.cxxflags-str;
          passthru = (args.passthru or { }) // {
            straylight-target = name;
          };
        }
      );

    __functor = _self: mkDerivation;
  };
}

# ────────────────────────────────────────────────────────────────────────────
#                          // arm -> x86 targets //
# ────────────────────────────────────────────────────────────────────────────

// lib.optionalAttrs platform.is-arm {
  # Reverse: aarch64 → x86_64
  x86-64 = rec {
    name = "x86-64";
    arch = "x86_64";
    target-gpu = gpu.sm_120;
    pkgs = final.pkgsCross.gnu64;

    mkDerivation =
      args:
      pkgs.stdenv.mkDerivation (
        args
        // turing-registry.attrs
        // {
          NIX_CFLAGS_COMPILE = (args.NIX_CFLAGS_COMPILE or "") + " " + turing-registry.cxxflags-str;
          passthru = (args.passthru or { }) // {
            straylight-target = name;
          };
        }
      );

    __functor = _self: mkDerivation;
  };
}
