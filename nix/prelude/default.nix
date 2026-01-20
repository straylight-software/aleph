# nix/prelude/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                             // the prelude //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games, in early
#     graphics programs and military experimentation with cranial
#     jacks. Cyberspace. A consensual hallucination experienced daily
#     by billions of legitimate operators, in every nation, by children
#     being taught mathematical concepts... A graphic representation
#     of data abstracted from the banks of every computer in the human
#     system. Unthinkable complexity. Lines of light ranged in the
#     nonspace of the mind, clusters and constellations of data.
#     Like city lights, receding...
#
#                                                         — Neuromancer
#
# The Weyl Prelude as an overlay. A membrane between your code and the
# nixpkgs substrate. You write lisp-case, structured, version-pinned.
# The membrane translates.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   straylight.prelude           the functional library + builders
#   straylight.platform          platform detection
#   straylight.gpu               GPU architecture metadata
#   straylight.turing-registry   the non-negotiable build flags
#
#   aleph.eval             evaluate typed Haskell expressions
#   aleph.import           import typed modules as attrsets
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final: prev:
let
  inherit (prev) lib;

  # ──────────────────────────────────────────────────────────────────────────
  #                              // imports //
  # ──────────────────────────────────────────────────────────────────────────

  platform = import ./platform.nix { inherit lib final; };
  gpu = import ./gpu.nix { inherit lib; };
  turing-registry = import ./turing-registry.nix { inherit lib platform; };

  toolchain = import ./toolchain.nix {
    inherit
      lib
      final
      platform
      turing-registry
      ;
  };

  stdenv = import ./stdenv.nix {
    inherit
      lib
      final
      platform
      turing-registry
      toolchain
      ;
  };

  cross = import ./cross.nix {
    inherit
      lib
      final
      platform
      gpu
      turing-registry
      ;
  };

  translations = import ./translations.nix { inherit lib; };
  versions = import ./versions.nix { };
  license = import ./license.nix { inherit lib; };
  schemas = import ./schemas.nix { inherit lib; };

  # The functional prelude (lists, strings, attrs, etc.)
  functions = import ./functions { inherit lib; };

  # Language-specific namespaces
  languages = import ./languages { inherit lib final; };

  # Builders (fetch, render, script)
  builders = import ./builders { inherit lib final turing-registry; };

  # ──────────────────────────────────────────────────────────────────────────
  #                              // assembly //
  # ──────────────────────────────────────────────────────────────────────────

  prelude =
    functions
    // languages
    // builders
    // {
      inherit
        platform
        gpu
        turing-registry
        stdenv
        cross
        schemas
        ;
      inherit versions license;
      inherit (translations) translate-attrs;
    };

in
{
  # ──────────────────────────────────────────────────────────────────────────
  #                            // straylight namespace //
  # ──────────────────────────────────────────────────────────────────────────

  straylight = {
    inherit
      prelude
      platform
      gpu
      turing-registry
      stdenv
      cross
      ;
    inherit (toolchain) llvm;
    inherit versions license;
    inherit (translations) translate-attrs;

    # Toolchain paths for downstream consumers
    toolchain = toolchain.paths;

    # Introspection
    info = {
      inherit platform;
      inherit (turing-registry) cflags cxxflags attrs;

      gcc = toolchain.gcc-info;

      nvidia = lib.optionalAttrs platform.is-linux {
        sdk = toolchain.nvidia-sdk;
        arch = toolchain.default-gpu-arch;
      };

      stdenvs = builtins.attrNames stdenv;
      cross-targets = builtins.attrNames cross;
    };
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                          // legacy aliases //
  # ──────────────────────────────────────────────────────────────────────────
  # For migration. These will be removed.

  aleph-naughtenv = stdenv.default;
  aleph-naughtenv-static = stdenv.static or stdenv.default;
  aleph-naughtenv-musl = stdenv.clang-musl-dynamic or stdenv.default;
  aleph-naughtenv-musl-static = stdenv.portable or stdenv.default;
  aleph-naughtenv-nvidia = stdenv.nvidia or null;
  straylight-cross = cross;
  aleph-naughtenv-info = final.straylight.info;
}
