#!/usr/bin/env bash
# nix/lib/scripts/buck2-build.bash
#
# Buck2 build phase for Nix builds.
#
# Environment variables (set by Nix):
#   $buck2Target - Buck2 target to build

runHook preBuild

export HOME=$TMPDIR
buck2 build "$buck2Target" --show-full-output 2>&1 | tee build.log

runHook postBuild
