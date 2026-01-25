-- nix/checks/scripts/test-cpp-hello.dhall
--
-- Test C++ Hello World compilation
-- Environment variables are injected by render.dhall-with-vars

let cppHello : Text = env:CPP_HELLO as Text

in ''
#!/usr/bin/env bash
# Test C++ Hello World compilation

echo "Testing: $CXX --version"
$CXX --version || c++ --version || g++ --version

echo ""
echo "Creating test program..."
cp "${cppHello}" hello.cpp

echo "Compiling..."
c++ -o hello hello.cpp

echo "Running..."
./hello
''
