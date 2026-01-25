-- nix/checks/scripts/test-rust-hello.dhall
--
-- Test Rust Hello World compilation
-- Environment variables are injected by render.dhall-with-vars

let rustHello : Text = env:RUST_HELLO as Text

in ''
#!/usr/bin/env bash
# Test Rust Hello World compilation

echo "Testing: rustc --version"
rustc --version

echo ""
echo "Testing: rustc can compile hello world"
cp "${rustHello}" hello.rs

rustc -o hello hello.rs
./hello
''
