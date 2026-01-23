# nix/overlays/straylight-nix.nix
#
# straylight nix overlay - provides builtins.wasm-enabled nix binary
#
# Usage in devshell:
#   pkgs.straylight.nix  # The nix binary with builtins.wasm
#
# The wrapper adds --no-eval-cache by default to avoid stale derivation
# path issues during development.
#
{ inputs }:
let
  mkStraylightNixPackages =
    pkgs: system:
    let
      straylightNixPkgs = inputs.nix.packages.${system} or { };
      unwrappedNix = straylightNixPkgs.nix or null;

      # Wrap nix to add --no-eval-cache by default
      # This avoids stale derivation path issues during development
      wrappedNix =
        if unwrappedNix == null then
          null
        else
          pkgs.writeShellScriptBin "nix" ''
            exec ${unwrappedNix}/bin/nix --no-eval-cache "$@"
          '';

      # Full wrapper that includes all nix subcommands and man pages
      nixWrapper =
        if unwrappedNix == null then
          null
        else
          pkgs.symlinkJoin {
            name = "straylight-nix";
            paths = [
              wrappedNix
              unwrappedNix
            ];
            # wrappedNix comes first, so its bin/nix takes precedence
            postBuild = ''
              # Remove the unwrapped nix binary, keep the wrapper
              rm $out/bin/nix
              cp ${wrappedNix}/bin/nix $out/bin/nix
            '';
          };
    in
    {
      # The main nix binary with builtins.wasm support + --no-eval-cache
      nix = nixWrapper;

      # Unwrapped version if someone needs it
      nix-unwrapped = unwrappedNix;

      # Man pages
      nix-man = straylightNixPkgs.nix-man or null;
    };
in
{
  flake.overlays.straylight-nix = final: _prev: {
    straylight = (_prev.straylight or { }) // {
      nix = mkStraylightNixPackages final final.stdenv.hostPlatform.system;
    };
  };
}
