# nix/packages/aleph-exec.nix
#
# DEPRECATED: Use aleph-build instead (Aleph-1)
#
# This is a compatibility wrapper that re-exports aleph-build.
# The old aleph-exec based on the phase-action system is condemned.
#
{
  lib,
  callPackage,
  ...
}:

# Just call aleph-build
callPackage ./aleph-build.nix { }
