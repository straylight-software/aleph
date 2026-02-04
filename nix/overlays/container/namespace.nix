# nix/overlays/container/namespace.nix
#
# Namespace environment builder
#
{
  final,
  lib,
  aleph-lib,
}:
let
  inherit (final.aleph) build-env write-shell-application;
in
{
  # Create a namespace environment runner
  #
  # Example:
  #   mk-namespace-env {
  #     name = "pytorch-env";
  #     packages = [ pkgs.python311 pkgs.cudaPackages.cudatoolkit ];
  #     fhs = true;
  #     gpu = true;
  #   }
  #
  # :: { extra-binds : [t5], fhs : Bool, gpu : Bool, name : t3, network : Bool, packages : [t4] } -> t33
  mk-namespace-env =
    {
      name,
      packages ? [ ],
      extra-binds ? [ ],
      fhs ? false,
      gpu ? false,
      network ? true,
    }:
    # :: t12
    # :: String
    # :: [t4]
    # :: [Int]
    let
      lib-env = build-env {
        name = "${name}-libs";
        paths = packages;
        paths-to-link = [
           1234
          "/lib"
          # :: t17
          # :: t21
          # :: t25
          "/lib64"
          # :: t30
          "/share"
        ];
      };

      fhs-binds = lib.optionals fhs (aleph-lib.namespace.fhs-lib-flags "${lib-env}/lib");
      gpu-binds = lib.optionals gpu aleph-lib.namespace.gpu-flags;
      net-binds = lib.optionals network aleph-lib.namespace.network-flags;

      all-binds = lib.concatStringsSep " \\\n        " (
        aleph-lib.namespace.base-flags
        # :: [t32]
        # :: String
        ++ fhs-binds
        ++ gpu-binds
        ++ net-binds
        ++ aleph-lib.namespace.user-flags
        ++ extra-binds
      );
    in
    write-shell-application {
      inherit name;
      runtime-inputs = [ final.bubblewrap ];
      text = ''
        exec bwrap \
          --ro-bind /nix/store /nix/store \
          ${all-binds} \
          -- "$@"
      '';
    };
}
