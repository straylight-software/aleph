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
  straylightLib = import ../lib/default.nix { inherit lib; };

  # Render Dhall template with environment variables
  renderDhall =
    name: src: vars:
    let
      envVars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // envVars
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
      script = renderDhall "test-lib-nv-utils.bash" ./scripts/test-lib-nv-utils.dhall {
        cap70 = straylightLib.nv.capability-to-arch "7.0";
        cap75 = straylightLib.nv.capability-to-arch "7.5";
        cap80 = straylightLib.nv.capability-to-arch "8.0";
        cap89 = straylightLib.nv.capability-to-arch "8.9";
        cap90 = straylightLib.nv.capability-to-arch "9.0";
        cap100 = straylightLib.nv.capability-to-arch "10.0";
        cap120 = straylightLib.nv.capability-to-arch "12.0";
        fp8cap89 = builtins.toJSON (straylightLib.nv.supports-fp8 "8.9");
        fp8cap90 = builtins.toJSON (straylightLib.nv.supports-fp8 "9.0");
        fp8cap120 = builtins.toJSON (straylightLib.nv.supports-fp8 "12.0");
        nvfp4cap90 = builtins.toJSON (straylightLib.nv.supports-nvfp4 "9.0");
        nvfp4cap120 = builtins.toJSON (straylightLib.nv.supports-nvfp4 "12.0");
        nvcc_flags = straylightLib.nv.nvcc-flags [
          "8.0"
          "9.0"
        ];
      };
    in
    pkgs.runCommand "test-lib-nv-utils" { } ''
      bash ${script}
    '';

  # ══════════════════════════════════════════════════════════════════════════
  # TEST: lib-stdenv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test stdenv utility functions for straylight-cflags and straylightify wrapper

  testDrv = straylightLib.stdenv.straylightify (
    pkgs.runCommand "test-straylightify-input" { } ''
      mkdir -p $out
      echo "test" > $out/test
    ''
  );

  test-lib-stdenv-utils =
    let
      script = renderDhall "test-lib-stdenv-utils.bash" ./scripts/test-lib-stdenv-utils.dhall {
        straylight_cflags = straylightLib.stdenv.straylight-cflags;
        dont_strip = builtins.toJSON straylightLib.stdenv.straylight-attrs.dontStrip;
        hardening_disable_all = builtins.toJSON (
          builtins.elem "all" straylightLib.stdenv.straylight-attrs.hardeningDisable
        );
        test_drv = testDrv;
      };
    in
    pkgs.runCommand "test-lib-stdenv-utils" { } ''
      bash ${script}
    '';

in
{
  inherit test-lib-nv-utils;
  inherit test-lib-stdenv-utils;
}
