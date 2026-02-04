# ==============================================================================
#                         ALEPH MODULE COMPILATION TESTS
#
#  Verifies that all Aleph.* modules compile successfully.
#  This catches import errors, type errors, and missing dependencies.
#
#  We use GHC's --make mode which handles dependency ordering automatically.
# ==============================================================================
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # Use lib functions with lisp-case local aliases
  # :: t15
  # :: t16
  # :: [String] -> [String] -> String -> String
  # :: Int | Float | Bool | Path | String -> String
  # :: t17
  # :: t18
  join = lib.concatStringsSep;
  map-attrs' = lib.mapAttrs';
  replace = builtins.replaceStrings;
  to-string = builtins.toString;
  to-upper = lib.toUpper;
  when-attr = lib.optionalAttrs;
# :: t19

  inherit (pkgs.aleph) run-command;

  # :: t20 -> t21 -> t22 -> t39
  # Get script source and GHC from the overlay
  # script-lib = library sources (Aleph/*)
  # :: t34
  # script-src = executable sources (*.hs scripts)
  script-lib = pkgs.aleph.script.lib;
  # :: t31
  # :: String
  inherit (pkgs.aleph.script) ghc;

  # Render Dhall template with environment variables
  render-dhall =
    name: src: vars:
    let
      env-vars = map-attrs' (
        k: v:
        let
          # :: [t37]
          name = to-upper (replace [ "-" ] [ "_" ] k);
          value = to-string v;
        in
        {
          inherit name value;
        }
      ) vars;
    in
    run-command name
      (
        {
          native-build-inputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      # :: t44
      )
      ''
        # :: [t42]
        # :: t9
        dhall text --file ${src} > $out
      '';

  # ==============================================================================
  # TEST: aleph-modules
  # ==============================================================================
  # Compile all Aleph.* modules using GHC's --make mode.
  # This automatically handles dependency ordering and verifies everything compiles.

  # Use run-command since GHC uses system CC which may be gcc (avoid aleph stdenv clang flags)
  test-aleph-modules =
    run-command "test-aleph-modules"
      {
        # :: ["vfio-bind"]
        native-build-inputs = [ ghc ];
        src = script-lib;
      }
      ''
        ${builtins.readFile ./scripts/test-aleph-modules.bash}
        mkdir -p $out
        echo "SUCCESS" > $out/SUCCESS
        echo "All Aleph modules compiled successfully" >> $out/SUCCESS
      '';

  # ==============================================================================
  # TEST: aleph-compiled-scripts
  # ==============================================================================
  # Verify all compiled scripts in aleph.script.compiled build successfully

  script-names = [
    "vfio-bind"
    "vfio-unbind"
    "vfio-list"
    "crane-inspect"
    "crane-pull"
    "unshare-run"
    "unshare-gpu"
    "fhs-run"
    "gpu-run"
    "isospin-run"
    "isospin-build"
    "cloud-hypervisor-run"
    "cloud-hypervisor-gpu"
  ];

  script-check-lines = join "\n" (
    map (name: ''
      echo "  Checking ${name}..."
      if [ ! -x "${pkgs.aleph.script.compiled.${name}}/bin/${name}" ]; then
        echo "FAILED: ${name} not found or not executable"
        exit 1
      fi
      echo "    ${pkgs.aleph.script.compiled.${name}}/bin/${name}"
    '') script-names
  );

  test-aleph-compiled-scripts =
    let
      script =
        render-dhall "test-aleph-compiled-scripts.bash" ./scripts/test-aleph-compiled-scripts.dhall
          {
            script-checks = script-check-lines;
          };
    in
    run-command "test-aleph-compiled-scripts" { } ''
      bash ${script}
    '';

in
# Only run on Linux (Aleph.Nix has FFI bindings that may need Linux)
when-attr (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit
    test-aleph-modules
    test-aleph-compiled-scripts
    ;
}
