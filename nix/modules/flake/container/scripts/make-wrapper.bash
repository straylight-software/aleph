#!/usr/bin/env bash
# Create a wrapped binary with config and PATH
set -euo pipefail

mkdir -p "$out/bin"
makeWrapper "@sourceBin@" "$out/bin/@binName@" \
  --set CONFIG_FILE "@configFile@" \
  --prefix PATH : "@binPath@"
