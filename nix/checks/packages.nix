# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH-NAUGHT PACKAGE TESTS                          ║
# ║                                                                            ║
# ║  Tests for packages exposed by aleph-naught.                               ║
# ║  Ensures packages are properly built and usable.                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # ══════════════════════════════════════════════════════════════════════════
  # TEST: mdspan-installation
  # ══════════════════════════════════════════════════════════════════════════
  # Verify that mdspan headers are properly installed and can be used
  # to compile a C++23 program using std::mdspan

  test-mdspan-installation = pkgs.stdenv.mkDerivation {
    name = "test-mdspan-installation";

    src = pkgs.writeTextDir "test.cpp" (builtins.readFile ./test-sources/mdspan-test.cpp);

    nativeBuildInputs = [
      pkgs.gcc15
      pkgs.mdspan
    ];

    buildPhase = ''
      echo "Building mdspan test program..."
      g++ -std=c++23 -I${pkgs.mdspan}/include test.cpp -o test
    '';

    doCheck = true;
    checkPhase = ''
      echo "Running mdspan test..."
      ./test
      echo "✓ mdspan test passed"
    '';

    installPhase = ''
      mkdir -p $out
      echo "SUCCESS" > $out/SUCCESS
      echo "mdspan C++23 headers work correctly" >> $out/SUCCESS
    '';

    meta = {
      description = "Test that mdspan C++23 headers are properly installed and usable";
    };
  };

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: nvidia-sdk-structure (Linux-only)
  # ══════════════════════════════════════════════════════════════════════════
  # Verify that the NVIDIA SDK has the expected structure and critical headers

  test-nvidia-sdk-structure =
    pkgs.runCommand "test-nvidia-sdk-structure"
      {
        nativeBuildInputs = [ pkgs.nvidia-sdk ];
      }
      (
        pkgs.replaceVars ./scripts/test-nvidia-sdk-structure.bash {
          nvidiaSdk = pkgs.nvidia-sdk;
        }
      );

in
{
  # Always include mdspan test
  inherit test-mdspan-installation;

  # Only include NVIDIA SDK test on Linux
}
// lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit test-nvidia-sdk-structure;
}
