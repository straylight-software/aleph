# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH-NAUGHT LIBRARY TESTS                          ║
# ║                                                                            ║
# ║  Tests for pure library functions in nix/lib/default.nix.                  ║
# ║  These test NVIDIA GPU utilities and stdenv utilities.                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  lib,
  ...
}:
let
  straylightLib = import ../lib/default.nix { inherit lib; };

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: lib-nv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test NVIDIA GPU utility functions for capability/arch conversion and feature detection

  test-lib-nv-utils = pkgs.runCommand "test-lib-nv-utils" { } ''
    echo "Testing NVIDIA GPU utility functions..."

    # Test capability-to-arch conversions
    echo "Testing capability-to-arch..."

    # Volta
    if [ "${straylightLib.nv.capability-to-arch "7.0"}" != "volta" ]; then
      echo "✗ FAILED: capability-to-arch 7.0"
      exit 1
    fi
    echo "✓ 7.0 -> volta"

    # Turing
    if [ "${straylightLib.nv.capability-to-arch "7.5"}" != "turing" ]; then
      echo "✗ FAILED: capability-to-arch 7.5"
      exit 1
    fi
    echo "✓ 7.5 -> turing"

    # Ampere (8.0)
    if [ "${straylightLib.nv.capability-to-arch "8.0"}" != "ampere" ]; then
      echo "✗ FAILED: capability-to-arch 8.0"
      exit 1
    fi
    echo "✓ 8.0 -> ampere"

    # Ada
    if [ "${straylightLib.nv.capability-to-arch "8.9"}" != "ada" ]; then
      echo "✗ FAILED: capability-to-arch 8.9"
      exit 1
    fi
    echo "✓ 8.9 -> ada"

    # Hopper
    if [ "${straylightLib.nv.capability-to-arch "9.0"}" != "hopper" ]; then
      echo "✗ FAILED: capability-to-arch 9.0"
      exit 1
    fi
    echo "✓ 9.0 -> hopper"

    # Blackwell (10.0)
    if [ "${straylightLib.nv.capability-to-arch "10.0"}" != "blackwell" ]; then
      echo "✗ FAILED: capability-to-arch 10.0"
      exit 1
    fi
    echo "✓ 10.0 -> blackwell"

    # Blackwell (12.0)
    if [ "${straylightLib.nv.capability-to-arch "12.0"}" != "blackwell" ]; then
      echo "✗ FAILED: capability-to-arch 12.0"
      exit 1
    fi
    echo "✓ 12.0 -> blackwell"

    # Test supports-fp8
    echo ""
    echo "Testing supports-fp8..."

    # Ada (8.9) should NOT support FP8
    if [ "${builtins.toJSON (straylightLib.nv.supports-fp8 "8.9")}" != "false" ]; then
      echo "✗ FAILED: supports-fp8 8.9 should be false"
      exit 1
    fi
    echo "✓ 8.9 does NOT support FP8"

    # Hopper (9.0) should support FP8
    if [ "${builtins.toJSON (straylightLib.nv.supports-fp8 "9.0")}" != "true" ]; then
      echo "✗ FAILED: supports-fp8 9.0 should be true"
      exit 1
    fi
    echo "✓ 9.0 supports FP8"

    # Blackwell (12.0) should support FP8
    if [ "${builtins.toJSON (straylightLib.nv.supports-fp8 "12.0")}" != "true" ]; then
      echo "✗ FAILED: supports-fp8 12.0 should be true"
      exit 1
    fi
    echo "✓ 12.0 supports FP8"

    # Test supports-nvfp4
    echo ""
    echo "Testing supports-nvfp4..."

    # Hopper (9.0) should NOT support NVFP4
    if [ "${builtins.toJSON (straylightLib.nv.supports-nvfp4 "9.0")}" != "false" ]; then
      echo "✗ FAILED: supports-nvfp4 9.0 should be false"
      exit 1
    fi
    echo "✓ 9.0 does NOT support NVFP4"

    # Blackwell (12.0) should support NVFP4
    if [ "${builtins.toJSON (straylightLib.nv.supports-nvfp4 "12.0")}" != "true" ]; then
      echo "✗ FAILED: supports-nvfp4 12.0 should be true"
      exit 1
    fi
    echo "✓ 12.0 supports NVFP4"

    # Test nvcc-flags
    echo ""
    echo "Testing nvcc-flags..."

    flags="${
      straylightLib.nv.nvcc-flags [
        "8.0"
        "9.0"
      ]
    }"
    expected="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"
    if [ "$flags" != "$expected" ]; then
      echo "✗ FAILED: nvcc-flags"
      echo "  Got: $flags"
      echo "  Expected: $expected"
      exit 1
    fi
    echo "✓ nvcc-flags generates correct flags"

    mkdir -p $out
    echo "SUCCESS" > $out/SUCCESS
    echo "All NVIDIA GPU utility function tests passed" >> $out/SUCCESS
  '';

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: lib-stdenv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test stdenv utility functions for straylight-cflags and straylightify wrapper

  test-lib-stdenv-utils = pkgs.runCommand "test-lib-stdenv-utils" { } ''
    echo "Testing stdenv utility functions..."

    # Test straylight-cflags contains expected flags
    echo "Testing straylight-cflags..."

    cflags="${straylightLib.stdenv.straylight-cflags}"
    echo "straylight-cflags: $cflags"

    # Check for -O2
    if ! echo "$cflags" | grep -q -- "-O2"; then
      echo "✗ FAILED: straylight-cflags missing -O2"
      exit 1
    fi
    echo "✓ Contains -O2"

    # Check for -g3
    if ! echo "$cflags" | grep -q -- "-g3"; then
      echo "✗ FAILED: straylight-cflags missing -g3"
      exit 1
    fi
    echo "✓ Contains -g3"

    # Check for -gdwarf-5
    if ! echo "$cflags" | grep -q -- "-gdwarf-5"; then
      echo "✗ FAILED: straylight-cflags missing -gdwarf-5"
      exit 1
    fi
    echo "✓ Contains -gdwarf-5"

    # Check for -fno-omit-frame-pointer
    if ! echo "$cflags" | grep -q -- "-fno-omit-frame-pointer"; then
      echo "✗ FAILED: straylight-cflags missing -fno-omit-frame-pointer"
      exit 1
    fi
    echo "✓ Contains -fno-omit-frame-pointer"

    # Check for -D_FORTIFY_SOURCE=0
    if ! echo "$cflags" | grep -q -- "-D_FORTIFY_SOURCE=0"; then
      echo "✗ FAILED: straylight-cflags missing -D_FORTIFY_SOURCE=0"
      exit 1
    fi
    echo "✓ Contains -D_FORTIFY_SOURCE=0"

    # Check for -std=c++23
    if ! echo "$cflags" | grep -q -- "-std=c++23"; then
      echo "✗ FAILED: straylight-cflags missing -std=c++23"
      exit 1
    fi
    echo "✓ Contains -std=c++23"

    # Test straylight-attrs
    echo ""
    echo "Testing straylight-attrs..."

    # Test that dontStrip is true
    if [ "${builtins.toJSON straylightLib.stdenv.straylight-attrs.dontStrip}" != "true" ]; then
      echo "✗ FAILED: straylight-attrs.dontStrip is not true"
      exit 1
    fi
    echo "✓ dontStrip is true"

    # Test that hardeningDisable contains "all"
    if [ "${builtins.toJSON (builtins.elem "all" straylightLib.stdenv.straylight-attrs.hardeningDisable)}" != "true" ]; then
      echo "✗ FAILED: straylight-attrs.hardeningDisable does not contain 'all'"
      exit 1
    fi
    echo "✓ hardeningDisable contains 'all'"

    # Test straylightify wrapper
    echo ""
    echo "Testing straylightify wrapper..."

    # Create a simple derivation and straylightify it
    testDrv="${
      straylightLib.stdenv.straylightify (
        pkgs.runCommand "test-straylightify-input" { } ''
          mkdir -p $out
          echo "test" > $out/test
        ''
      )
    }"

    # Verify it builds (if it throws, the test fails)
    if [ ! -f "$testDrv/test" ]; then
      echo "✗ FAILED: straylightify wrapper broke the derivation"
      exit 1
    fi
    echo "✓ straylightify wrapper works"

    mkdir -p $out
    echo "SUCCESS" > $out/SUCCESS
    echo "All stdenv utility function tests passed" >> $out/SUCCESS
  '';

in
{
  inherit test-lib-nv-utils;
  inherit test-lib-stdenv-utils;
}
