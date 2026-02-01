#!/usr/bin/env bash
# Test GHC Hello World compilation

echo "Creating test program..."
cp "@ghcHello@" Main.hs

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
