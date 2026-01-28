#!/usr/bin/env bash
# Install phase for Buck2 builds (auto-detect mode)
# Environment variables:
#   TARGET_NAME - name of the target to find

runHook preInstall

mkdir -p "$out/bin"

# Auto-detect: copy executables from buck-out
find buck-out/v2/gen -type f -executable -name "${TARGET_NAME}*" | head -1 | xargs -I{} cp {} "$out/bin/"

runHook postInstall
