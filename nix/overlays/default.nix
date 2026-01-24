# overlays/default.nix
#
# Composes all aleph-naught overlays:
#   - packages: mdspan, etc.
#   - llvm-git: LLVM from git with SM120 Blackwell support
#   - nvidia-sdk: cuda-packages_13 bundled, no fallbacks
#   - nvidia-sdk-ngc: CUDA/cuDNN/TensorRT extracted from NGC containers
#   - nvidia-sdk-packages: autopatchelf'd nvidia-sdk, nvidia-tritonserver, etc.
#   - prelude: straylight.prelude, straylight.stdenv, straylight.cross, etc.
#   - container: straylight.container.mkNamespaceEnv, unshare-run, etc.
#   - script: straylight.script.gen-wrapper, straylight.script.check, etc.
#   - ghc-wasm: GHC WASM backend for builtins.wasm integration
#   - straylight-nix: nix binary with builtins.wasm support
#   - libmodern: pkgs.libmodern.fmt, pkgs.libmodern.abseil-cpp, etc.
#
inputs:
let
  inherit (inputs.nixpkgs) lib;

  # Package overlay - adds packages from overlays/packages/
  packages-overlay = final: _prev: {
    mdspan = final.callPackage ./packages/mdspan.nix { };
  };

  # Individual overlays - each is a function: final: prev: { ... }
  llvm-git-overlay = import ./llvm-git.nix inputs;
  nvidia-sdk-overlay = import ./nixpkgs-nvidia-sdk.nix;
  nvidia-sdk-ngc-overlay = import ./nvidia-sdk;
  nvidia-sdk-packages-overlay = import ./nvidia-sdk/packages.nix;
  prelude-overlay = import ../prelude;
  container-overlay = import ./container;
  script-overlay = import ./script.nix;
  ghc-wasm-overlay = (import ./ghc-wasm.nix { inherit inputs; }).flake.overlays.ghc-wasm;
  straylight-nix-overlay =
    (import ./straylight-nix.nix { inherit inputs; }).flake.overlays.straylight-nix;
  libmodern-overlay = import ./libmodern;
in
{
  flake.overlays = {
    packages = packages-overlay;
    llvm-git = llvm-git-overlay;
    nvidia-sdk = nvidia-sdk-overlay;
    nvidia-sdk-ngc = nvidia-sdk-ngc-overlay;
    nvidia-sdk-packages = nvidia-sdk-packages-overlay;
    prelude = prelude-overlay;
    container = container-overlay;
    script = script-overlay;
    ghc-wasm = ghc-wasm-overlay;
    straylight-nix = straylight-nix-overlay;
    libmodern = libmodern-overlay;

    # Composed default overlay
    # Order matters:
    #   1. script must come before nvidia-sdk-packages (needs straylight.script.compiled)
    #   2. nvidia-sdk-ngc must come before nvidia-sdk-packages (needs container FOD)
    default = lib.composeManyExtensions [
      packages-overlay
      llvm-git-overlay
      nvidia-sdk-overlay
      prelude-overlay
      container-overlay
      script-overlay
      nvidia-sdk-ngc-overlay
      nvidia-sdk-packages-overlay
      ghc-wasm-overlay
      straylight-nix-overlay
      libmodern-overlay
    ];
  };
}
