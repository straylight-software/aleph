#!/usr/bin/env bash
# Install phase for Buck2 builds (explicit output mode)
# Environment variables:
#   OUTPUT_PATH - path within buck-out to copy

runHook preInstall

mkdir -p "$out/bin"

# Copy the specified output
cp buck-out/v2/gen/*/"$OUTPUT_PATH" "$out/bin/"

runHook postInstall
