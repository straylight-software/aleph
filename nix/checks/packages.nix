# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH PACKAGE TESTS                          ║
# ║                                                                            ║
# ║  Tests for packages exposed by aleph.                               ║
# ║  Ensures packages are properly built and usable.                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # ────────────────────────────────────────────────────────────────────────────
  # Lisp-case functions from prelude
  # ────────────────────────────────────────────────────────────────────────────
  inherit (pkgs.aleph.prelude)
    get'
    map-attrs'
    read-file
    replace
    to-string
    to-upper
    when-attr
    ;

  inherit (pkgs.aleph) run-command stdenv;

  # ────────────────────────────────────────────────────────────────────────────
  # Lisp-case aliases for pkgs.* functions
  # ────────────────────────────────────────────────────────────────────────────
  write-text-dir = get' "writeTextDir" pkgs;

  # Haskell packages alias
  haskell-packages = get' "haskellPackages" pkgs;

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      env-vars = map-attrs' (k: v: {
        name = to-upper (replace [ "-" ] [ "_" ] k);
        value = to-string v;
      }) vars;
    in
    run-command name
      (
        {
          native-build-inputs = [ haskell-packages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: mdspan-installation
  # ══════════════════════════════════════════════════════════════════════════
  # Verify that mdspan headers are properly installed and can be used
  # to compile a C++23 program using std::mdspan

  test-mdspan-installation = stdenv.default {
    name = "test-mdspan-installation";

    src = write-text-dir "test.cpp" (read-file ./test-sources/mdspan-test.cpp);

    native-build-inputs = [
      pkgs.gcc15
      pkgs.mdspan
    ];

    build-phase = ''
      echo "Building mdspan test program..."
      g++ -std=c++23 -I${pkgs.mdspan}/include test.cpp -o test
    '';

    do-check = true;
    check-phase = ''
      echo "Running mdspan test..."
      ./test
      echo "✓ mdspan test passed"
    '';

    install-phase = ''
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
    let
      script = render-dhall "test-nvidia-sdk-structure.bash" ./scripts/test-nvidia-sdk-structure.dhall {
        nvidia-sdk = pkgs.nvidia-sdk;
      };
    in
    run-command "test-nvidia-sdk-structure"
      {
        native-build-inputs = [ pkgs.nvidia-sdk ];
      }
      ''
        bash ${script}
      '';

in
{
  # Always include mdspan test
  inherit test-mdspan-installation;

  # Only include NVIDIA SDK test on Linux
}
// when-attr (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit test-nvidia-sdk-structure;
}
