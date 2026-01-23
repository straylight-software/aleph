# nix/overlays/container/oci.nix
#
# OCI image extraction and runners
#
{ final }:
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
    final.stdenvNoCC.mkDerivation {
      inherit name;
      __contentAddressed = true;

      nativeBuildInputs = [
        final.crane
        final.gnutar
        final.gzip
      ];

      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      outputHash = hash;

      SSL_CERT_FILE = "${final.cacert}/etc/ssl/certs/ca-bundle.crt";

      buildCommand = ''
        mkdir -p $out
        crane export --platform ${platform} "${ref}" - | tar -xf - -C $out
      '';

      meta = {
        description = "OCI container image rootfs extracted to Nix store";
      };
    };

  # Run OCI images in a namespace — compiled Haskell, not bash
  inherit (final.straylight.script.compiled) oci-run;
}
