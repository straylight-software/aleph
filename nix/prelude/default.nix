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
# The Aleph Prelude as an overlay. A membrane between your code and the
# nixpkgs substrate. You write lisp-case, structured, version-pinned.
# The membrane translates.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   aleph.prelude           the functional library + builders
#   aleph.platform          platform detection
#   aleph.gpu               GPU architecture metadata
#   aleph.turing-registry   the non-negotiable build flags
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

  # Typed foundation (Dhall -> Nix at build time)
  types = import ./types { pkgs = final; };

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

  # ──────────────────────────────────────────────────────────────────────────
  #                           // typed wrappers //
  # ──────────────────────────────────────────────────────────────────────────
  # These wrap nixpkgs builders with translation. The boundary.

  inherit (translations) translate-attrs;

  # run-command: like pkgs.runCommand but accepts lisp-case attrs
  run-command =
    name: attrs: script:
    final.runCommand name (translate-attrs attrs) script;

  # write-shell-application: like pkgs.writeShellApplication but lisp-case
  write-shell-application = args: final.writeShellApplication (translate-attrs args);

  # build-env: like pkgs.buildEnv but accepts lisp-case attrs
  build-env = args: final.buildEnv (translate-attrs args);

  # fixed-output-derivation: typed wrapper for FODs
  #
  # Creates a fixed-output derivation with proper content-addressing.
  # Uses stdenvNoCC since FODs don't need a compiler.
  #
  # Required attrs:
  #   name         - derivation name
  #   hash         - expected output hash (SRI format, e.g. "sha256-...")
  #   build-script - shell script to produce output
  #
  # Optional attrs:
  #   hash-algo    - hash algorithm (default: "sha256")
  #   hash-mode    - "flat" or "recursive" (default: "recursive")
  #   native-build-inputs, etc. - passed through with translation
  #
  # Example:
  #   aleph.fixed-output-derivation {
  #     name = "my-fetched-thing";
  #     hash = "sha256-abc123...";
  #     native-build-inputs = [ pkgs.curl ];
  #     build-script = ''
  #       curl -o $out https://example.com/file
  #     '';
  #   }
  #
  fixed-output-derivation =
    {
      name,
      hash,
      build-script,
      hash-algo ? "sha256",
      hash-mode ? "recursive",
      ...
    }@args:
    let
      # Extract FOD-specific attrs, pass rest through
      rest = builtins.removeAttrs args [
        "name"
        "hash"
        "build-script"
        "hash-algo"
        "hash-mode"
      ];
    in
    final.stdenvNoCC.mkDerivation (
      translate-attrs (
        rest
        // {
          inherit name;
          # FOD configuration
          output-hash-algo = hash-algo;
          output-hash-mode = hash-mode;
          output-hash = hash;
          # The build script
          build-command = build-script;
        }
      )
    );

  prelude =
    functions
    // languages
    // builders
    // {
      inherit
        types
        platform
        gpu
        turing-registry
        stdenv
        cross
        schemas
        run-command
        write-shell-application
        build-env
        fixed-output-derivation
        ;
      inherit versions license;
    };

in
{
  # ──────────────────────────────────────────────────────────────────────────
  #                            // aleph namespace //
  # ──────────────────────────────────────────────────────────────────────────

  aleph = {
    inherit
      prelude
      types
      platform
      gpu
      turing-registry
      stdenv
      cross
      run-command
      write-shell-application
      build-env
      fixed-output-derivation
      ;
    inherit (toolchain) llvm;
    inherit versions license;

    # Toolchain paths for downstream consumers
    toolchain = toolchain.paths;

    # Introspection
    info = {
      inherit platform;
      inherit (turing-registry) cflags cxxflags attrs;

      gcc = toolchain.gcc-info;

      nvidia = functions.when platform.is-linux {
        sdk = toolchain.nvidia-sdk;
        arch = toolchain.default-gpu-arch;
      };

      stdenvs = functions.keys stdenv;
      cross-targets = functions.keys cross;
    };
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                          // legacy aliases //
  # ──────────────────────────────────────────────────────────────────────────
  # For migration. These will be removed.

  alephenv = stdenv.default;
  alephenv-static = stdenv.static or stdenv.default;
  alephenv-musl = stdenv.clang-musl-dynamic or stdenv.default;
  alephenv-musl-static = stdenv.portable or stdenv.default;
  alephenv-nvidia = if stdenv ? nvidia then stdenv.nvidia else null;
  aleph-cross = cross;
  alephenv-info = final.aleph.info;
}
