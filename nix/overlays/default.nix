# overlays/default.nix
#
# Composes all aleph overlays:
#   - packages: mdspan, etc.
#   - llvm-git: LLVM from git with SM120 Blackwell support
#   - nvidia-sdk: cuda-packages_13 bundled, no fallbacks
#   - nvidia-sdk-ngc: CUDA/cuDNN/TensorRT extracted from NGC containers
#   - nvidia-sdk-packages: autopatchelf'd nvidia-sdk, nvidia-tritonserver, etc.
#   - prelude: aleph.prelude, aleph.stdenv, aleph.cross, etc.
#   - container: aleph.container.mkNamespaceEnv, unshare-run, etc.
#   - script: aleph.script.gen-wrapper, aleph.script.check, etc.
#   - ghc-wasm: GHC WASM backend for builtins.wasm integration
#   - straylight-nix: nix binary with builtins.wasm support
#   - libmodern: pkgs.libmodern.fmt, pkgs.libmodern.abseil-cpp, etc.
#   - armitage: armitage.proxy (witness proxy for build-time fetches)
#   - haskell: GHC 9.12 overrides (ghc-source-gen, grapesy stack)
#
inputs:
let
  inherit (inputs.nixpkgs) lib;

  # Lisp-case aliases for lib functions
  compose-many-extensions = lib.composeManyExtensions;

  # Package overlay - adds packages from overlays/packages/
  packages-overlay =
    final: _prev:
    let
      call-package = final.callPackage;
    in
    {
      mdspan = call-package ./packages/mdspan.nix { };
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
  armitage-overlay = import ./armitage.nix;
  haskell-overlay = import ./haskell.nix { inherit inputs; };
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
    armitage = armitage-overlay;
    haskell = haskell-overlay;

    # Composed default overlay
    # Order matters:
    #   1. prelude-overlay FIRST so all others can use aleph.*
    #   2. script must come before nvidia-sdk-packages (needs aleph.script.compiled)
    #   3. nvidia-sdk-ngc must come before nvidia-sdk-packages (needs container FOD)
    #
    # Note: Nix overlays use lazy fixed-point evaluation, so prelude can access
    # final.llvm-git even though llvm-git-overlay is listed later.
    default = compose-many-extensions [
      prelude-overlay
      packages-overlay
      llvm-git-overlay
      nvidia-sdk-overlay
      container-overlay
      script-overlay
      nvidia-sdk-ngc-overlay
      nvidia-sdk-packages-overlay
      ghc-wasm-overlay
      straylight-nix-overlay
      libmodern-overlay
      haskell-overlay # must come before armitage (armitage uses hs-pkgs)
      armitage-overlay
    ];
  };
}
