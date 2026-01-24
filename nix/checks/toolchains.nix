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
  inherit (pkgs) runCommand;

  # ─────────────────────────────────────────────────────────────────────────
  # Helper: create a smoke test derivation
  # ─────────────────────────────────────────────────────────────────────────
  mkSmokeTest =
    {
      name,
      description,
      nativeBuildInputs ? [ ],
      buildInputs ? [ ],
      testScript,
    }:
    runCommand "test-toolchain-${name}"
      {
        inherit nativeBuildInputs buildInputs;
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
        ${testScript}
        echo ""
        echo "✓ ${name} smoke test passed"
        touch $out
      '';

in
{
  # ─────────────────────────────────────────────────────────────────────────
  # GHC Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  ghc-version = mkSmokeTest {
    name = "ghc-version";
    description = "GHC compiler reports version";
    nativeBuildInputs = [ prelude.ghc.pkg ];
    testScript = builtins.readFile ./scripts/test-ghc-version.bash;
  };

  ghc-hello-world = mkSmokeTest {
    name = "ghc-hello-world";
    description = "GHC can compile and run Hello World";
    nativeBuildInputs = [ prelude.ghc.pkg ];
    testScript = pkgs.replaceVars ./scripts/test-ghc-hello.bash {
      ghcHello = ./test-sources/ghc-hello.hs;
    };
  };

  ghc-with-packages = mkSmokeTest {
    name = "ghc-with-packages";
    description = "ghcWithPackages provides requested packages";
    nativeBuildInputs = [
      (prelude.ghc.pkgs'.ghcWithPackages (
        p: with p; [
          text
          bytestring
        ]
      ))
    ];
    testScript = pkgs.replaceVars ./scripts/test-ghc-packages.bash {
      ghcText = ./test-sources/ghc-text.hs;
    };
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Python Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  python-version = mkSmokeTest {
    name = "python-version";
    description = "Python interpreter reports version";
    nativeBuildInputs = [ prelude.python.pkg ];
    testScript = builtins.readFile ./scripts/test-python-version.bash;
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Rust Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  rust-version = mkSmokeTest {
    name = "rust-version";
    description = "Rust compiler reports version";
    nativeBuildInputs = [ prelude.rust.pkg ];
    testScript = pkgs.replaceVars ./scripts/test-rust-hello.bash {
      rustHello = ./test-sources/rust-hello.rs;
    };
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Lean Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  lean-version = mkSmokeTest {
    name = "lean-version";
    description = "Lean prover reports version";
    nativeBuildInputs = [ prelude.lean.pkg ];
    testScript = builtins.readFile ./scripts/test-lean-version.bash;
  };

  # ─────────────────────────────────────────────────────────────────────────
  # C++ Toolchain (via stdenv)
  # ─────────────────────────────────────────────────────────────────────────

  cpp-hello-world = mkSmokeTest {
    name = "cpp-hello-world";
    description = "C++ compiler can compile Hello World";
    nativeBuildInputs = [ pkgs.stdenv.cc ];
    testScript = pkgs.replaceVars ./scripts/test-cpp-hello.bash {
      cppHello = ./test-sources/cpp-hello.cpp;
    };
  };
}
