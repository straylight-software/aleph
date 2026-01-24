-- nix/checks/scripts/test-lib-nv-utils.dhall
--
-- Test script for NVIDIA GPU utility functions
-- Environment variables are injected by render.dhall-with-vars

let cap70 : Text = env:CAP70 as Text
let cap75 : Text = env:CAP75 as Text
let cap80 : Text = env:CAP80 as Text
let cap89 : Text = env:CAP89 as Text
let cap90 : Text = env:CAP90 as Text
let cap100 : Text = env:CAP100 as Text
let cap120 : Text = env:CAP120 as Text
let fp8cap89 : Text = env:FP8CAP89 as Text
let fp8cap90 : Text = env:FP8CAP90 as Text
let fp8cap120 : Text = env:FP8CAP120 as Text
let nvfp4cap90 : Text = env:NVFP4CAP90 as Text
let nvfp4cap120 : Text = env:NVFP4CAP120 as Text
let nvccFlags : Text = env:NVCC_FLAGS as Text

in ''
#!/usr/bin/env bash
# Test NVIDIA GPU utility functions

echo "Testing NVIDIA GPU utility functions..."

# Test capability-to-arch conversions
echo "Testing capability-to-arch..."

# Volta
if [ "${cap70}" != "volta" ]; then
	echo "x FAILED: capability-to-arch 7.0"
	exit 1
fi
echo "v 7.0 -> volta"

# Turing
if [ "${cap75}" != "turing" ]; then
	echo "x FAILED: capability-to-arch 7.5"
	exit 1
fi
echo "v 7.5 -> turing"

# Ampere (8.0)
if [ "${cap80}" != "ampere" ]; then
	echo "x FAILED: capability-to-arch 8.0"
	exit 1
fi
echo "v 8.0 -> ampere"

# Ada
if [ "${cap89}" != "ada" ]; then
	echo "x FAILED: capability-to-arch 8.9"
	exit 1
fi
echo "v 8.9 -> ada"

# Hopper
if [ "${cap90}" != "hopper" ]; then
	echo "x FAILED: capability-to-arch 9.0"
	exit 1
fi
echo "v 9.0 -> hopper"

# Blackwell (10.0)
if [ "${cap100}" != "blackwell" ]; then
	echo "x FAILED: capability-to-arch 10.0"
	exit 1
fi
echo "v 10.0 -> blackwell"

# Blackwell (12.0)
if [ "${cap120}" != "blackwell" ]; then
	echo "x FAILED: capability-to-arch 12.0"
	exit 1
fi
echo "v 12.0 -> blackwell"

# Test supports-fp8
echo ""
echo "Testing supports-fp8..."

# Ada (8.9) should NOT support FP8
if [ "${fp8cap89}" != "false" ]; then
	echo "x FAILED: supports-fp8 8.9 should be false"
	exit 1
fi
echo "v 8.9 does NOT support FP8"

# Hopper (9.0) should support FP8
if [ "${fp8cap90}" != "true" ]; then
	echo "x FAILED: supports-fp8 9.0 should be true"
	exit 1
fi
echo "v 9.0 supports FP8"

# Blackwell (12.0) should support FP8
if [ "${fp8cap120}" != "true" ]; then
	echo "x FAILED: supports-fp8 12.0 should be true"
	exit 1
fi
echo "v 12.0 supports FP8"

# Test supports-nvfp4
echo ""
echo "Testing supports-nvfp4..."

# Hopper (9.0) should NOT support NVFP4
if [ "${nvfp4cap90}" != "false" ]; then
	echo "x FAILED: supports-nvfp4 9.0 should be false"
	exit 1
fi
echo "v 9.0 does NOT support NVFP4"

# Blackwell (12.0) should support NVFP4
if [ "${nvfp4cap120}" != "true" ]; then
	echo "x FAILED: supports-nvfp4 12.0 should be true"
	exit 1
fi
echo "v 12.0 supports NVFP4"

# Test nvcc-flags
echo ""
echo "Testing nvcc-flags..."

flags="${nvccFlags}"
expected="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"
if [ "$flags" != "$expected" ]; then
	echo "x FAILED: nvcc-flags"
	echo "  Got: $flags"
	echo "  Expected: $expected"
	exit 1
fi
echo "v nvcc-flags generates correct flags"

mkdir -p "$out"
echo "SUCCESS" >"$out/SUCCESS"
echo "All NVIDIA GPU utility function tests passed" >>"$out/SUCCESS"
''
