# nix/overlays/container/oci.nix
#
# OCI image extraction and runners
#
{ final, lib }:
let
  # Import prelude for translate-attrs
  translations = import ../../prelude/translations.nix { inherit lib; };
  inherit (translations) translate-attrs;
in
{
  # Extract OCI image to Nix store (content-addressed)
  #
  # Example:
  #   mk-oci-rootfs {
  #     name = "pytorch-rootfs";
  #     ref = "nvcr.io/nvidia/pytorch:25.01-py3";
  #     hash = "sha256-...";  # empty string for first build
  #   }
  #
  mk-oci-rootfs =
    {
      name,
      ref,
      hash ? "",
      platform ? "linux/amd64",
    }:
    final.stdenvNoCC.mkDerivation (
      translate-attrs {
        inherit name;
        __contentAddressed = true;

        native-build-inputs = [
          final.crane
          final.gnutar
          final.gzip
        ];

        SSL_CERT_FILE = "${final.cacert}/etc/ssl/certs/ca-bundle.crt";

        meta = {
          description = "OCI container image rootfs extracted to Nix store";
        };
      }
      // {
        # NOTE: FOD attrs are nixpkgs API, quoted
        "outputHashAlgo" = "sha256";
        "outputHashMode" = "recursive";
        "outputHash" = hash;

        "buildCommand" = ''
          mkdir -p $out
          crane export --platform ${platform} "${ref}" - | tar -xf - -C $out
        '';
      }
    );

  # Run OCI images in bwrap/unshare namespace — compiled Haskell, not bash
  unshare-run = final.straylight.script.compiled.unshare-run;
}
