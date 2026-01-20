# nix/overlays/straylight-nix.nix
#
# straylight nix overlay - provides builtins.wasm-enabled nix binary
#
# Usage in devshell:
#   pkgs.straylight.nix  # The nix binary with builtins.wasm
#
{ inputs }:
let
  mkStraylightNixPackages =
    system:
    let
      straylightNixPkgs = inputs.nix.packages.${system} or { };
    in
    {
      # The main nix binary with builtins.wasm support
      nix = straylightNixPkgs.nix or null;

      # Man pages
      nix-man = straylightNixPkgs.nix-man or null;
    };
in
{
  flake.overlays.straylight-nix = final: _prev: {
    straylight = (_prev.straylight or { }) // {
      nix = mkStraylightNixPackages final.stdenv.hostPlatform.system;
    };
  };
}
