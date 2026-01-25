-- nix/checks/scripts/test-lib-stdenv-utils.dhall
--
-- Test stdenv utility functions
-- Environment variables are injected by render.dhall-with-vars

let straylightCflags : Text = env:STRAYLIGHT_CFLAGS as Text
let dontStrip : Text = env:DONT_STRIP as Text
let hardeningDisableAll : Text = env:HARDENING_DISABLE_ALL as Text
let testDrv : Text = env:TEST_DRV as Text

in ''
#!/usr/bin/env bash
# Test stdenv utility functions

echo "Testing stdenv utility functions..."

# Test straylight-cflags contains expected flags
echo "Testing straylight-cflags..."

cflags="${straylightCflags}"
echo "straylight-cflags: $cflags"

# Check for -O2
if ! echo "$cflags" | grep -q -- "-O2"; then
	echo "x FAILED: straylight-cflags missing -O2"
	exit 1
fi
echo "v Contains -O2"

# Check for -g3
if ! echo "$cflags" | grep -q -- "-g3"; then
	echo "x FAILED: straylight-cflags missing -g3"
	exit 1
fi
echo "v Contains -g3"

# Check for -gdwarf-5
if ! echo "$cflags" | grep -q -- "-gdwarf-5"; then
	echo "x FAILED: straylight-cflags missing -gdwarf-5"
	exit 1
fi
echo "v Contains -gdwarf-5"

# Check for -fno-omit-frame-pointer
if ! echo "$cflags" | grep -q -- "-fno-omit-frame-pointer"; then
	echo "x FAILED: straylight-cflags missing -fno-omit-frame-pointer"
	exit 1
fi
echo "v Contains -fno-omit-frame-pointer"

# Check for -D_FORTIFY_SOURCE=0
if ! echo "$cflags" | grep -q -- "-D_FORTIFY_SOURCE=0"; then
	echo "x FAILED: straylight-cflags missing -D_FORTIFY_SOURCE=0"
	exit 1
fi
echo "v Contains -D_FORTIFY_SOURCE=0"

# Check for -std=c++23
if ! echo "$cflags" | grep -q -- "-std=c++23"; then
	echo "x FAILED: straylight-cflags missing -std=c++23"
	exit 1
fi
echo "v Contains -std=c++23"

# Test straylight-attrs
echo ""
echo "Testing straylight-attrs..."

# Test that dontStrip is true
if [ "${dontStrip}" != "true" ]; then
	echo "x FAILED: straylight-attrs.dontStrip is not true"
	exit 1
fi
echo "v dontStrip is true"

# Test that hardeningDisable contains "all"
if [ "${hardeningDisableAll}" != "true" ]; then
	echo "x FAILED: straylight-attrs.hardeningDisable does not contain 'all'"
	exit 1
fi
echo "v hardeningDisable contains 'all'"

# Test straylightify wrapper
echo ""
echo "Testing straylightify wrapper..."

# Verify it builds (if it throws, the test fails)
if [ ! -f "${testDrv}/test" ]; then
	echo "x FAILED: straylightify wrapper broke the derivation"
	exit 1
fi
echo "v straylightify wrapper works"

mkdir -p "$out"
echo "SUCCESS" >"$out/SUCCESS"
echo "All stdenv utility function tests passed" >>"$out/SUCCESS"
''
