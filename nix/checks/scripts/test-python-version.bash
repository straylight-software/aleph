#!/usr/bin/env bash
# Test Python version and basic arithmetic

echo "Testing: python3 --version"
python3 --version

echo ""
echo "Testing: python3 -c 'print(1+1)'"
OUTPUT=$(python3 -c 'print(1+1)')
if [ "$OUTPUT" != "2" ]; then
	echo "ERROR: Expected 2, got $OUTPUT"
	exit 1
fi
echo "Python arithmetic works correctly"
