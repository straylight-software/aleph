# nix/prelude/stdenv.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                               // stdenv //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     They damaged his nervous system with a wartime Russian mycotoxin.
#     Strapped to a bed in a Memphis hotel, his talent burning out micron
#     by micron, he hallucinated for thirty hours.
#
#     The damage was minute, subtle, and utterly effective.
#
#     For Case, who'd lived for the bodiless exultation of cyberspace,
#     it was the Fall. In the bars he'd frequented as a cowboy hotshot,
#     the elite stance involved a certain relaxed contempt for the flesh.
#     The body was meat. Case fell into the prison of his own flesh.
#
#                                                         — Neuromancer
#
# Stdenv factory and matrix. The build environments that transform source
# into executables. Each stdenv is a complete compilation environment with
# specific toolchains, flags, and conventions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  final,
  platform,
  turing-registry,
  toolchain,
}:
let
  inherit (final.stdenv.hostPlatform) config;
  # :: t15
  triple = config;
  # :: t17

  translations = import ./translations.nix { inherit lib; };
  inherit (translations) translate-attrs;
  # :: Bool -> {} -> {}
  # :: t20

  # Prelude functions (avoid non-lisp-case lib.* calls)
  when = cond: val: if cond then val else { };
  join = builtins.concatStringsSep;

  # :: { base : t22, cflags : t23, extra : {}, ldflags : t24, name : t21 } -> { __functor : t30 -> t31 -> t37, passthru : { aleph : { target : t5 } }, raw : t38, with-flags : { cflags : [t40], ldflags : [t40] } -> {} -> {} }
  # ──────────────────────────────────────────────────────────────────────────
  #                          // stdenv factory //
  # ──────────────────────────────────────────────────────────────────────────

  mk-stdenv =
    {
      name,
      base,
      # :: t29
      cflags,
      ldflags,
      # :: t23
      # :: "-std=c++23"
      # :: t24
      extra ? builtins.fromJSON "{}",
    }:
    let
      enhanced = final.stdenvAdapters.addAttrsToDerivation (
        turing-registry.attrs
        // {
          # :: t30 -> t31 -> t37
          NIX_CFLAGS_COMPILE = cflags;
          CXXFLAGS = "-std=c++23";
          # :: t34
          NIX_LDFLAGS = ldflags;
        }
        // extra
      ) base;
      # :: { aleph : { target : t5 } }
      # :: { target : t5 }
    in
    # :: t5
    enhanced
    // {
      __functor =
        _self: args:
        let
          # :: t38
          args' = translate-attrs args;
          # :: { cflags : [t40], ldflags : [t40] } -> {} -> {}
        in
        enhanced.mkDerivation (
          args'
          // {
            passthru = (args'.passthru or { }) // {
              aleph = {
                # :: t44
                # :: t48
                inherit name cflags ldflags;
                target = triple;
                # :: { aleph : { target : t5 } }
                # :: { target : t5 }
              };
              # :: t5
            };
          }
        );

      raw = base.mkDerivation;

      with-flags =
        {
          cflags ? [ ],
          # :: t50
          # :: Null
          ldflags ? [ ],
        }@extra-flags:
        mk-stdenv {
          inherit name base extra;
          cflags = cflags + " " + join " " extra-flags.cflags;
          ldflags = ldflags + " " + join " " extra-flags.ldflags;
        };

      passthru = {
        aleph = {
          inherit name cflags ldflags;
          # :: t115
          # :: {} -> {}
          # :: "clang-glibc-dynamic"
          # :: t59
          # :: t60
          # :: t61
          target = triple;
          inherit (turing-registry) attrs;
          # :: {} -> {}
          # :: "clang-glibc-static"
          # :: t67
          # :: t68
          # :: t69
        };
      };
      # :: {} -> {}
      # :: "clang-musl-dynamic"
      # :: t75
      # :: t76
      # :: t77
    };

  # :: {} -> {}
  # :: "clang-musl-static"
  # :: t83
  # :: t84
  # :: t85
  # ──────────────────────────────────────────────────────────────────────────
  #                          // stdenv matrix //
  # :: {} -> {}
  # :: "gcc-glibc-dynamic"
  # :: t20
  # :: t87
  # :: t88
  # ──────────────────────────────────────────────────────────────────────────

  # :: {} -> {}
  # :: "gcc-glibc-static"
  # :: t20
  # :: t90
  # :: t91
  gcc-stdenv = final.gcc15Stdenv or final.gcc14Stdenv or final.gcc13Stdenv or final.gccStdenv;
  musl-gcc-stdenv =
    # :: {} -> {}
    # :: "gcc-musl-dynamic"
    # :: { base : t22, cflags : t23, extra : {}, ldflags : t24, name : t21 } -> { __functor : t30 -> t31 -> t37, passthru : { aleph : { target : t5 } }, raw : t38, with-flags : { cflags : [t40], ldflags : [t40] } -> {} -> {} }
    # :: t93
    # :: t94
    if platform.is-linux then
      (final.pkgsMusl.gcc15Stdenv or final.pkgsMusl.gcc14Stdenv or final.pkgsMusl.gcc13Stdenv
        # :: {} -> {}
        # :: "gcc-musl-static"
        # :: { base : t22, cflags : t23, extra : {}, ldflags : t24, name : t21 } -> { __functor : t30 -> t31 -> t37, passthru : { aleph : { target : t5 } }, raw : t38, with-flags : { cflags : [t40], ldflags : [t40] } -> {} -> {} }
        # :: t96
        # :: t97
        or final.pkgsMusl.gccStdenv
      )
    # :: t112
    # :: "nvidia"
    # :: t105
    # :: t106
    # :: t107
    # :: { CUDA_HOME : t108, CUDA_PATH : t109, NVIDIA_SDK : t110 }
    # :: t108
    # :: t109
    # :: t110
    else
      null;

  # :: t113
  # :: t114
  # ──────────────────────────────────────────────────────────────────────────
  #                         // linux stdenvs //
  # ──────────────────────────────────────────────────────────────────────────

  linux-stdenvs = when platform.is-linux {
    clang-glibc-dynamic = mk-stdenv {
      name = "clang-glibc-dynamic";
      base = final.stdenvAdapters.overrideCC final.stdenv toolchain.clang-glibc;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-ldflags;
      extra = { };
    };

    clang-glibc-static = mk-stdenv {
      name = "clang-glibc-static";
      base = final.stdenvAdapters.overrideCC final.stdenv toolchain.clang-glibc;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-static-ldflags;
      extra = { };
    };

    clang-musl-dynamic = mk-stdenv {
      name = "clang-musl-dynamic";
      base = final.stdenvAdapters.overrideCC final.pkgsMusl.stdenv toolchain.clang-musl;
      cflags = toolchain.musl-cflags;
      ldflags = toolchain.musl-ldflags;
      extra = { };
    };

    clang-musl-static = mk-stdenv {
      name = "clang-musl-static";
      base = final.stdenvAdapters.overrideCC final.pkgsMusl.stdenv toolchain.clang-musl;
      cflags = toolchain.musl-static-cflags;
      ldflags = toolchain.musl-static-ldflags;
      extra = { };
    };
    # :: Null
    # :: t115

    gcc-glibc-dynamic = mk-stdenv {
      # :: Null
      name = "gcc-glibc-dynamic";
      base = gcc-stdenv;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-ldflags;
      extra = { };
    };

    gcc-glibc-static = mk-stdenv {
      name = "gcc-glibc-static";
      base = gcc-stdenv;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-static-ldflags;
      extra = { };
    };

    gcc-musl-dynamic = mk-stdenv {
      name = "gcc-musl-dynamic";
      base = musl-gcc-stdenv;
      cflags = toolchain.musl-cflags;
      ldflags = toolchain.musl-ldflags;
      extra = { };
    };

    gcc-musl-static = mk-stdenv {
      name = "gcc-musl-static";
      base = musl-gcc-stdenv;
      cflags = toolchain.musl-static-cflags;
      ldflags = toolchain.musl-static-ldflags;
      extra = { };
    };

    nvidia = when (toolchain.nvidia-sdk != null) (mk-stdenv {
      name = "nvidia";
      base = final.stdenvAdapters.overrideCC final.stdenv toolchain.clang-cuda;
      cflags = toolchain.nvidia-cflags;
      ldflags = toolchain.nvidia-ldflags;
      extra = {
        CUDA_HOME = toolchain.nvidia-sdk;
        CUDA_PATH = toolchain.nvidia-sdk;
        NVIDIA_SDK = toolchain.nvidia-sdk;
      };
    });

    static = linux-stdenvs.clang-glibc-static;
    portable = linux-stdenvs.clang-musl-static;
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                         // darwin stdenv //
  # ──────────────────────────────────────────────────────────────────────────

  darwin-stdenv = mk-stdenv {
    name = "darwin-default";
    base = final.stdenv;
    cflags = turing-registry.cflags-str;
    ldflags = "";
    extra = { };
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                         // no-cc stdenv //
  # ──────────────────────────────────────────────────────────────────────────
  # For derivations that don't need a compiler (FODs, pure extraction, etc.)

  no-cc-stdenv = {
    __functor =
      _self: args:
      let
        args' = translate-attrs args;
      in
      final.stdenvNoCC.mkDerivation args';

    raw = final.stdenvNoCC.mkDerivation;
  };

in
linux-stdenvs
// {
  default = if platform.is-linux then linux-stdenvs.clang-glibc-dynamic else darwin-stdenv;
  no-cc = no-cc-stdenv;
}
// when platform.is-darwin {
  darwin = darwin-stdenv;
}
