# nix/modules/nixos/_index.nix
#
# Index of all NixOS modules (:: NixOSModule)
#
# The directory is the kind signature.
#
{
  nv-driver = import ./nv-driver.nix;
}
