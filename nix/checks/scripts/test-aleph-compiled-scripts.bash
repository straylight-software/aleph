#!/usr/bin/env bash
# Verify all compiled Aleph scripts

echo "Verifying compiled Aleph scripts..."
echo ""

# Check that key binaries exist and are executable
@scriptChecks@

mkdir -p "$out"
echo "SUCCESS" >"$out/SUCCESS"
echo "All compiled Aleph scripts verified" >>"$out/SUCCESS"
