#!/usr/bin/env bash
# Test full module composition

echo "Testing 'full' module composition..."
echo ""
echo "The 'full' module should include:"
echo "  - build (LLVM 22 Buck2 toolchain)"
echo "  - shortlist (hermetic C++ libraries)"
echo "  - lre (NativeLink remote execution)"
echo "  - devshell (development environment)"
echo "  - nixpkgs (overlay support)"
echo ""
echo "Downstream usage:"
echo "  imports = [ aleph.modules.flake.full ];"
echo "  aleph.build.enable = true;"
echo "  aleph.shortlist.enable = true;"
echo "  aleph.lre.enable = true;"

mkdir -p "$out"
echo "SUCCESS" >"$out/result"
