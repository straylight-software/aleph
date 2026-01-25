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
  triple = config;

  translations = import ./translations.nix { inherit lib; };
  inherit (translations) translate-attrs;

  # Prelude functions (avoid non-lisp-case lib.* calls)
  when = cond: val: if cond then val else { };
  join = builtins.concatStringsSep;

  # ──────────────────────────────────────────────────────────────────────────
  #                          // stdenv factory //
  # ──────────────────────────────────────────────────────────────────────────

  mk-stdenv =
    {
      name,
      base,
      cflags,
      ldflags,
      extra ? { },
    }:
    let
      enhanced = final.stdenvAdapters.addAttrsToDerivation (
        turing-registry.attrs
        // {
          NIX_CFLAGS_COMPILE = cflags;
          CXXFLAGS = "-std=c++23";
          NIX_LDFLAGS = ldflags;
        }
        // extra
      ) base;
    in
    enhanced
    // {
      __functor =
        _self: args:
        let
          args' = translate-attrs args;
        in
        enhanced.mkDerivation (
          args'
          // {
            passthru = (args'.passthru or { }) // {
              aleph = {
                inherit name cflags ldflags;
                target = triple;
              };
            };
          }
        );

      raw = base.mkDerivation;

      with-flags =
        {
          cflags ? [ ],
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
          target = triple;
          inherit (turing-registry) attrs;
        };
      };
    };

  # ──────────────────────────────────────────────────────────────────────────
  #                          // stdenv matrix //
  # ──────────────────────────────────────────────────────────────────────────

  gcc-stdenv = final.gcc15Stdenv or final.gcc14Stdenv or final.gcc13Stdenv or final.gccStdenv;
  musl-gcc-stdenv =
    if platform.is-linux then
      (final.pkgsMusl.gcc15Stdenv or final.pkgsMusl.gcc14Stdenv or final.pkgsMusl.gcc13Stdenv
        or final.pkgsMusl.gccStdenv
      )
    else
      null;

  # ──────────────────────────────────────────────────────────────────────────
  #                         // linux stdenvs //
  # ──────────────────────────────────────────────────────────────────────────

  linux-stdenvs = when platform.is-linux {
    clang-glibc-dynamic = mk-stdenv {
      name = "clang-glibc-dynamic";
      base = final.stdenvAdapters.overrideCC final.stdenv toolchain.clang-glibc;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-ldflags;
    };

    clang-glibc-static = mk-stdenv {
      name = "clang-glibc-static";
      base = final.stdenvAdapters.overrideCC final.stdenv toolchain.clang-glibc;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-static-ldflags;
    };

    clang-musl-dynamic = mk-stdenv {
      name = "clang-musl-dynamic";
      base = final.stdenvAdapters.overrideCC final.pkgsMusl.stdenv toolchain.clang-musl;
      cflags = toolchain.musl-cflags;
      ldflags = toolchain.musl-ldflags;
    };

    clang-musl-static = mk-stdenv {
      name = "clang-musl-static";
      base = final.stdenvAdapters.overrideCC final.pkgsMusl.stdenv toolchain.clang-musl;
      cflags = toolchain.musl-static-cflags;
      ldflags = toolchain.musl-static-ldflags;
    };

    gcc-glibc-dynamic = mk-stdenv {
      name = "gcc-glibc-dynamic";
      base = gcc-stdenv;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-ldflags;
    };

    gcc-glibc-static = mk-stdenv {
      name = "gcc-glibc-static";
      base = gcc-stdenv;
      cflags = toolchain.glibc-cflags;
      ldflags = toolchain.glibc-static-ldflags;
    };

    gcc-musl-dynamic = mk-stdenv {
      name = "gcc-musl-dynamic";
      base = musl-gcc-stdenv;
      cflags = toolchain.musl-cflags;
      ldflags = toolchain.musl-ldflags;
    };

    gcc-musl-static = mk-stdenv {
      name = "gcc-musl-static";
      base = musl-gcc-stdenv;
      cflags = toolchain.musl-static-cflags;
      ldflags = toolchain.musl-static-ldflags;
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
