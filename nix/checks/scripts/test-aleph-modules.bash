#!/usr/bin/env bash
# Compile all Aleph modules

echo "Compiling all Aleph modules..."
echo ""

# Create temp directory for build artifacts
mkdir -p build

# Use --make to compile all modules with automatic dependency resolution
# We compile the "top-level" modules that pull in everything else:
# - Aleph.Script.Tools (imports all tool wrappers)
# - Aleph.Script.Vm (imports Vfio, Oci, Config)
# - Aleph.Nix (imports Types, Value, FFI)
# - Aleph.Nix.Syntax (imports Derivation, CMake)

ghc --make -Wall -Wno-unused-imports \
  -hidir build -odir build \
  -i"$src" \
  "$src/Aleph/Script.hs" \
  "$src/Aleph/Script/Tools.hs" \
  "$src/Aleph/Script/Vm.hs" \
  "$src/Aleph/Script/Oci.hs" \
  "$src/Aleph/Nix.hs" \
  "$src/Aleph/Nix/Syntax.hs" \
  2>&1 || {
  echo ""
  echo "FAILED: Module compilation failed"
  exit 1
}

echo ""
echo "All Aleph modules compiled successfully"
