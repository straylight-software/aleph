#!/usr/bin/env bash
# Test GHC with packages

echo "Testing: ghc-pkg list text"
ghc-pkg list text

echo ""
echo "Testing: ghc-pkg list bytestring"
ghc-pkg list bytestring

echo ""
echo "Testing: compile with text import"
cp "@ghcText@" TestText.hs

ghc -o test-text TestText.hs
OUTPUT=$(./test-text)
if [ "$OUTPUT" != "5" ]; then
	echo "ERROR: Expected 5, got $OUTPUT"
	exit 1
fi
echo "Text import works correctly"
