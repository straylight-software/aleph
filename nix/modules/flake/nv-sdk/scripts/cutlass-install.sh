#!/usr/bin/env bash
# cutlass-install.sh - Install CUTLASS headers and tools

runHook preInstall

mkdir -p "$out/include"
cp -r include/cutlass "$out/include/"
cp -r include/cute "$out/include/"

# Tools and examples for reference
mkdir -p "$out/share/cutlass"
cp -r tools "$out/share/cutlass/"
cp -r examples "$out/share/cutlass/"
cp -r python "$out/share/cutlass/"

echo "@version@" >"$out/CUTLASS_VERSION"

runHook postInstall
