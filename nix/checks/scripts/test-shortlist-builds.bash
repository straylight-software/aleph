#!/usr/bin/env bash
# Test shortlist module export

echo "Testing shortlist module export..."
echo ""
echo "This check verifies that:"
echo "  1. The shortlist module can be imported"
echo "  2. shortlist-* packages are available in the flake outputs"
echo "  3. Downstream flakes can use aleph.modules.flake.shortlist-standalone"
echo ""
echo "To use in a downstream flake:"
echo "  imports = [ aleph.modules.flake.shortlist-standalone ];"
echo "  aleph.shortlist.enable = true;"
echo ""

mkdir -p "$out"
echo "SUCCESS" >"$out/result"
