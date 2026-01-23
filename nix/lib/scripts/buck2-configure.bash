#!/usr/bin/env bash
# nix/lib/scripts/buck2-configure.bash
#
# Buck2 configure phase for Nix builds.
# Writes .buckconfig.local and links prelude.
#
# Environment variables (set by Nix):
#   $buckconfigFile - Path to buckconfig file
#   $prelude        - Path to buck2-prelude

runHook preConfigure

# Write .buckconfig.local with Nix store paths
cp "$buckconfigFile" .buckconfig.local
chmod 644 .buckconfig.local

# Link prelude if not present
if [ ! -d "prelude" ] && [ ! -L "prelude" ]; then
	ln -sf "$prelude" prelude
fi

runHook postConfigure
