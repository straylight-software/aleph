#!/usr/bin/env bash
# Test C++ Hello World compilation

echo 'Testing: $CXX --version'
$CXX --version || c++ --version || g++ --version

echo ""
echo "Creating test program..."
cp "@cppHello@" hello.cpp

echo "Compiling..."
c++ -o hello hello.cpp

echo "Running..."
./hello
