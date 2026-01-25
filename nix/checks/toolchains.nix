# nix/checks/toolchains.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                            // toolchains //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'It was a place, and she was there, and she was herself.
#      She couldn't define it, exactly, but she knew it was real.'
#
#                                                         — Neuromancer
#
# Smoke tests for toolchain wrappers. Every toolchain exposed by the prelude
# must have a test that verifies it works on the target platform.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  pkgs,
  prelude,
}:
let
  inherit (pkgs) lib;
  inherit (pkgs.aleph) run-command;

  to-string = builtins.toString;

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      env-vars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (to-string v)
      ) vars;
    in
    run-command name
      (
        {
          native-build-inputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # ─────────────────────────────────────────────────────────────────────────
  # Helper: create a smoke test derivation
  # ─────────────────────────────────────────────────────────────────────────
  mk-smoke-test =
    {
      name,
      description,
      native-build-inputs ? [ ],
      build-inputs ? [ ],
      test-script,
    }:
    run-command "test-toolchain-${name}"
      {
        inherit native-build-inputs build-inputs;
        passthru = {
          inherit description;
        };
      }
      ''
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  Toolchain Smoke Test: ${name}"
        echo "║  ${description}"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        ${test-script}
        echo ""
        echo "✓ ${name} smoke test passed"
        touch $out
      '';

in
{
  # ─────────────────────────────────────────────────────────────────────────
  # GHC Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  ghc-version = mk-smoke-test {
    name = "ghc-version";
    description = "GHC compiler reports version";
    native-build-inputs = [ prelude.ghc.pkg ];
    test-script = builtins.readFile ./scripts/test-ghc-version.bash;
  };

  ghc-hello-world = mk-smoke-test {
    name = "ghc-hello-world";
    description = "GHC can compile and run Hello World";
    native-build-inputs = [ prelude.ghc.pkg ];
    test-script = ''
      bash ${
        render-dhall "test-ghc-hello.bash" ./scripts/test-ghc-hello.dhall {
          ghc-hello = ./test-sources/ghc-hello.hs;
        }
      }
    '';
  };

  ghc-with-packages = mk-smoke-test {
    name = "ghc-with-packages";
    description = "ghcWithPackages provides requested packages";
    native-build-inputs = [
      (prelude.ghc.pkgs'.ghcWithPackages (
        p: with p; [
          text
          bytestring
        ]
      ))
    ];
    test-script = ''
      bash ${
        render-dhall "test-ghc-packages.bash" ./scripts/test-ghc-packages.dhall {
          ghc-text = ./test-sources/ghc-text.hs;
        }
      }
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Python Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  python-version = mk-smoke-test {
    name = "python-version";
    description = "Python interpreter reports version";
    native-build-inputs = [ prelude.python.pkg ];
    test-script = builtins.readFile ./scripts/test-python-version.bash;
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Rust Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  rust-version = mk-smoke-test {
    name = "rust-version";
    description = "Rust compiler reports version";
    native-build-inputs = [ prelude.rust.pkg ];
    test-script = ''
      bash ${
        render-dhall "test-rust-hello.bash" ./scripts/test-rust-hello.dhall {
          rust-hello = ./test-sources/rust-hello.rs;
        }
      }
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Lean Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  lean-version = mk-smoke-test {
    name = "lean-version";
    description = "Lean prover reports version";
    native-build-inputs = [ prelude.lean.pkg ];
    test-script = builtins.readFile ./scripts/test-lean-version.bash;
  };

  # ─────────────────────────────────────────────────────────────────────────
  # C++ Toolchain (via stdenv)
  # ─────────────────────────────────────────────────────────────────────────

  cpp-hello-world = mk-smoke-test {
    name = "cpp-hello-world";
    description = "C++ compiler can compile Hello World";
    native-build-inputs = [ pkgs.stdenv.cc ];
    test-script = ''
      bash ${
        render-dhall "test-cpp-hello.bash" ./scripts/test-cpp-hello.dhall {
          cpp-hello = ./test-sources/cpp-hello.cpp;
        }
      }
    '';
  };
}
