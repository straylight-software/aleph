#!/usr/bin/env bash
# Configure phase for Buck2 builds
# Environment variables:
#   BUCKCONFIG_FILE - path to buckconfig file
#   BUCK2_PRELUDE   - path to buck2 prelude

runHook preConfigure

# Write .buckconfig.local with Nix store paths
cp "$BUCKCONFIG_FILE" .buckconfig.local

# Link prelude if needed
if [ ! -d "prelude" ] && [ ! -L "prelude" ]; then
  ln -s "$BUCK2_PRELUDE" prelude
fi

runHook postConfigure
