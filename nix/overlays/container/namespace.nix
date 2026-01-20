# nix/overlays/container/namespace.nix
#
# Namespace environment builder
#
{
  final,
  lib,
  straylight-lib,
}:
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
  mk-namespace-env =
    {
      name,
      packages ? [ ],
      extra-binds ? [ ],
      fhs ? false,
      gpu ? false,
      network ? true,
    }:
    let
      lib-env = final.buildEnv {
        name = "${name}-libs";
        paths = packages;
        pathsToLink = [
          "/lib"
          "/lib64"
          "/share"
        ];
      };

      fhs-binds = lib.optionals fhs (straylight-lib.namespace.fhs-lib-flags "${lib-env}/lib");
      gpu-binds = lib.optionals gpu straylight-lib.namespace.gpu-flags;
      net-binds = lib.optionals network straylight-lib.namespace.network-flags;

      all-binds = lib.concatStringsSep " \\\n        " (
        straylight-lib.namespace.base-flags
        ++ fhs-binds
        ++ gpu-binds
        ++ net-binds
        ++ straylight-lib.namespace.user-flags
        ++ extra-binds
      );
    in
    final.writeShellApplication {
      inherit name;
      runtimeInputs = [ final.bubblewrap ];
      text = ''
        exec bwrap \
          --ro-bind /nix/store /nix/store \
          ${all-binds} \
          -- "$@"
      '';
    };
}
