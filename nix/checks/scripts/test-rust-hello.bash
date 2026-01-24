#!/usr/bin/env bash
# Test Rust Hello World compilation

echo "Testing: rustc --version"
rustc --version

echo ""
echo "Testing: rustc can compile hello world"
cp "@rustHello@" hello.rs

rustc -o hello hello.rs
./hello
