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
    testScript = ''
      echo "Testing: ghc --version"
      ghc --version
      echo ""
      echo "Testing: ghc --print-libdir"
      ghc --print-libdir
    '';
  };

  ghc-hello-world = mkSmokeTest {
    name = "ghc-hello-world";
    description = "GHC can compile and run Hello World";
    nativeBuildInputs = [ prelude.ghc.pkg ];
    testScript = ''
      echo "Creating test program..."
      cat > Main.hs << 'EOF'
      module Main where
      main :: IO ()
      main = putStrLn "Hello from GHC!"
      EOF

      echo "Compiling..."
      ghc -o hello Main.hs

      echo "Running..."
      ./hello

      # Verify output
      OUTPUT=$(./hello)
      if [ "$OUTPUT" != "Hello from GHC!" ]; then
        echo "ERROR: Unexpected output: $OUTPUT"
        exit 1
      fi
    '';
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
    testScript = ''
      echo "Testing: ghc-pkg list text"
      ghc-pkg list text

      echo ""
      echo "Testing: ghc-pkg list bytestring"
      ghc-pkg list bytestring

      echo ""
      echo "Testing: compile with text import"
      cat > TestText.hs << 'EOF'
      {-# LANGUAGE OverloadedStrings #-}
      module Main where
      import qualified Data.Text as T
      main = print (T.length "hello")
      EOF

      ghc -o test-text TestText.hs
      OUTPUT=$(./test-text)
      if [ "$OUTPUT" != "5" ]; then
        echo "ERROR: Expected 5, got $OUTPUT"
        exit 1
      fi
      echo "Text import works correctly"
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Python Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  python-version = mkSmokeTest {
    name = "python-version";
    description = "Python interpreter reports version";
    nativeBuildInputs = [ prelude.python.pkg ];
    testScript = ''
      echo "Testing: python3 --version"
      python3 --version

      echo ""
      echo "Testing: python3 -c 'print(1+1)'"
      OUTPUT=$(python3 -c 'print(1+1)')
      if [ "$OUTPUT" != "2" ]; then
        echo "ERROR: Expected 2, got $OUTPUT"
        exit 1
      fi
      echo "Python arithmetic works correctly"
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Rust Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  rust-version = mkSmokeTest {
    name = "rust-version";
    description = "Rust compiler reports version";
    nativeBuildInputs = [ prelude.rust.pkg ];
    testScript = ''
      echo "Testing: rustc --version"
      rustc --version

      echo ""
      echo "Testing: rustc can compile hello world"
      cat > hello.rs << 'EOF'
      fn main() {
          println!("Hello from Rust!");
      }
      EOF

      rustc -o hello hello.rs
      ./hello
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Lean Toolchain
  # ─────────────────────────────────────────────────────────────────────────

  lean-version = mkSmokeTest {
    name = "lean-version";
    description = "Lean prover reports version";
    nativeBuildInputs = [ prelude.lean.pkg ];
    testScript = ''
      echo "Testing: lean --version"
      lean --version
    '';
  };

  # ─────────────────────────────────────────────────────────────────────────
  # C++ Toolchain (via stdenv)
  # ─────────────────────────────────────────────────────────────────────────

  cpp-hello-world = mkSmokeTest {
    name = "cpp-hello-world";
    description = "C++ compiler can compile Hello World";
    nativeBuildInputs = [ pkgs.stdenv.cc ];
    testScript = ''
      echo "Testing: $CXX --version"
      $CXX --version || c++ --version || g++ --version

      echo ""
      echo "Creating test program..."
      cat > hello.cpp << 'EOF'
      #include <iostream>
      int main() {
          std::cout << "Hello from C++!" << std::endl;
          return 0;
      }
      EOF

      echo "Compiling..."
      c++ -o hello hello.cpp

      echo "Running..."
      ./hello
    '';
  };
}
