# nix/modules/nixos/_index.nix
#
# Index of all NixOS modules (:: NixOSModule)
#
# The directory is the kind signature.
#
{
  armitage-proxy = import ./armitage-proxy.nix;
  lre = import ./lre.nix;
  nix-proxy = import ./nix-proxy.nix;
  nv-driver = import ./nv-driver.nix;
}
