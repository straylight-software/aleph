# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ALEPH LIBRARY TESTS                          ║
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
  # :: t9
  # :: t11
  aleph-lib = import ../lib/default.nix { inherit lib; };
  prelude-fns = import ../prelude/functions.nix { inherit lib; };
  inherit (pkgs.aleph) run-command;
  inherit (prelude-fns) to-string;
# :: t12 -> t13 -> t14 -> t35

  # Render Dhall template with environment variables
  # :: t30
  render-dhall =
    name: src: vars:
    let
      env-vars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (to-string v)
      ) vars;
    # :: [t33]
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

  # :: t79
  # ══════════════════════════════════════════════════════════════════════════
  # :: t75
  # :: t40
  # :: t42
  # :: t44
  # :: t46
  # :: t48
  # :: t50
  # :: t52
  # :: t56
  # :: t60
  # :: t64
  # :: t68
  # :: t72
  # :: t74
  # TEST: lib-nv-utils
  # ══════════════════════════════════════════════════════════════════════════
  # Test NVIDIA GPU utility functions for capability/arch conversion and feature detection

  test-lib-nv-utils =
    let
      script = render-dhall "test-lib-nv-utils.bash" ./scripts/test-lib-nv-utils.dhall {
        cap70 = aleph-lib.nv.capability-to-arch "7.0";
        cap75 = aleph-lib.nv.capability-to-arch "7.5";
        cap80 = aleph-lib.nv.capability-to-arch "8.0";
        cap89 = aleph-lib.nv.capability-to-arch "8.9";
        cap90 = aleph-lib.nv.capability-to-arch "9.0";
        cap100 = aleph-lib.nv.capability-to-arch "10.0";
        cap120 = aleph-lib.nv.capability-to-arch "12.0";
        fp8cap89 = builtins.toJSON (aleph-lib.nv.supports-fp8 "8.9");
        fp8cap90 = builtins.toJSON (aleph-lib.nv.supports-fp8 "9.0");
        fp8cap120 = builtins.toJSON (aleph-lib.nv.supports-fp8 "12.0");
        nvfp4cap90 = builtins.toJSON (aleph-lib.nv.supports-nvfp4 "9.0");
        nvfp4cap120 = builtins.toJSON (aleph-lib.nv.supports-nvfp4 "12.0");
        nvcc-flags = aleph-lib.nv.nvcc-flags [
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
  # Test stdenv utility functions for aleph-cflags and alephify wrapper

  alephified-drv = aleph-lib.stdenv.alephify (
    run-command "test-alephify-input" { } ''
      mkdir -p $out
      echo "test" > $out/test
    ''
  );

  test-lib-stdenv-utils =
    let
      script = render-dhall "test-lib-stdenv-utils.bash" ./scripts/test-lib-stdenv-utils.dhall {
        inherit (aleph-lib.stdenv) aleph-cflags;
        dont-strip = builtins.toJSON aleph-lib.stdenv.aleph-attrs."dontStrip";
        hardening-disable-all = builtins.toJSON (
          builtins.elem "all" aleph-lib.stdenv.aleph-attrs."hardeningDisable"
        );
        test-drv = alephified-drv;
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
