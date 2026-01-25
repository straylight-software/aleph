# nix/overlays/straylight-nix.nix
#
# straylight nix overlay - provides builtins.wasm-enabled nix binary
#
# Usage in devshell:
#   pkgs.aleph.nix  # The nix binary with builtins.wasm
#
# The wrapper adds --no-eval-cache by default to avoid stale derivation
# path issues during development.
#
{ inputs }:
let
  # Check if straylight-nix input is available
  has-straylight-nix = inputs ? nix;

  mk-straylight-nix-packages =
    pkgs: system:
    let
      straylight-nix-pkgs = inputs.nix.packages.${system};
      unwrapped-nix = straylight-nix-pkgs.nix;

      # Wrap nix to add --no-eval-cache by default
      # This avoids stale derivation path issues during development
      wrapped-nix = pkgs.writeShellApplication {
        name = "nix";
        "runtimeInputs" = [ ];
        text = ''
          exec ${unwrapped-nix}/bin/nix --no-eval-cache "$@"
        '';
      };

      # Full wrapper that includes all nix subcommands and man pages
      nix-wrapper = pkgs.symlinkJoin {
        name = "straylight-nix";
        paths = [
          wrapped-nix
          unwrapped-nix
        ];
        # wrapped-nix comes first, so its bin/nix takes precedence
        "postBuild" = ''
          # Remove the unwrapped nix binary, keep the wrapper
          rm $out/bin/nix
          cp ${wrapped-nix}/bin/nix $out/bin/nix
        '';
      };
    in
    {
      # The main nix binary with builtins.wasm support + --no-eval-cache
      nix = nix-wrapper;

      # Unwrapped version if someone needs it
      nix-unwrapped = unwrapped-nix;

      # Man pages
      nix-man = straylight-nix-pkgs.nix-man;
    };
in
{
  flake.overlays.straylight-nix =
    final: _prev:
    if has-straylight-nix then
      {
        aleph = (_prev.aleph or { }) // {
          nix = mk-straylight-nix-packages final final.stdenv.hostPlatform.system;
        };
      }
    else
      { };
}
