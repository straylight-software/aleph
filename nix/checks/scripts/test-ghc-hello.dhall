-- nix/checks/scripts/test-ghc-hello.dhall
--
-- Test GHC Hello World compilation
-- Environment variables are injected by render.dhall-with-vars

let ghcHello : Text = env:GHC_HELLO as Text

in ''
#!/usr/bin/env bash
# Test GHC Hello World compilation

echo "Creating test program..."
cp "${ghcHello}" Main.hs

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
''
