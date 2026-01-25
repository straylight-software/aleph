# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH-NAUGHT LIBRARY TESTS                          ║
# ║                                                                            ║
# ║  Tests for pure library functions in nix/lib/default.nix.                  ║
# ║  These test NVIDIA GPU utilities and stdenv utilities.                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  lib,
  ...
}:
let
  straylight-lib = import ../lib/default.nix { inherit lib; };
  prelude-fns = import ../prelude/functions.nix { inherit lib; };
  inherit (pkgs.straylight) run-command;
  inherit (prelude-fns) to-string;

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      env-vars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (to-string v)
      ) vars;
    in
    run-command name
      (
        {
          native-build-inputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: lib-nv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test NVIDIA GPU utility functions for capability/arch conversion and feature detection

  test-lib-nv-utils =
    let
      script = render-dhall "test-lib-nv-utils.bash" ./scripts/test-lib-nv-utils.dhall {
        cap70 = straylight-lib.nv.capability-to-arch "7.0";
        cap75 = straylight-lib.nv.capability-to-arch "7.5";
        cap80 = straylight-lib.nv.capability-to-arch "8.0";
        cap89 = straylight-lib.nv.capability-to-arch "8.9";
        cap90 = straylight-lib.nv.capability-to-arch "9.0";
        cap100 = straylight-lib.nv.capability-to-arch "10.0";
        cap120 = straylight-lib.nv.capability-to-arch "12.0";
        fp8cap89 = builtins.toJSON (straylight-lib.nv.supports-fp8 "8.9");
        fp8cap90 = builtins.toJSON (straylight-lib.nv.supports-fp8 "9.0");
        fp8cap120 = builtins.toJSON (straylight-lib.nv.supports-fp8 "12.0");
        nvfp4cap90 = builtins.toJSON (straylight-lib.nv.supports-nvfp4 "9.0");
        nvfp4cap120 = builtins.toJSON (straylight-lib.nv.supports-nvfp4 "12.0");
        nvcc-flags = straylight-lib.nv.nvcc-flags [
          "8.0"
          "9.0"
        ];
      };
    in
    run-command "test-lib-nv-utils" { } ''
      bash ${script}
    '';

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: lib-stdenv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test stdenv utility functions for straylight-cflags and straylightify wrapper

  straylightified-drv = straylight-lib.stdenv.straylightify (
    run-command "test-straylightify-input" { } ''
      mkdir -p $out
      echo "test" > $out/test
    ''
  );

  test-lib-stdenv-utils =
    let
      script = render-dhall "test-lib-stdenv-utils.bash" ./scripts/test-lib-stdenv-utils.dhall {
        straylight-cflags = straylight-lib.stdenv.straylight-cflags;
        dont-strip = builtins.toJSON straylight-lib.stdenv.straylight-attrs."dontStrip";
        hardening-disable-all = builtins.toJSON (
          builtins.elem "all" straylight-lib.stdenv.straylight-attrs."hardeningDisable"
        );
        test-drv = straylightified-drv;
      };
    in
    run-command "test-lib-stdenv-utils" { } ''
      bash ${script}
    '';

in
{
  inherit test-lib-nv-utils;
  inherit test-lib-stdenv-utils;
}
