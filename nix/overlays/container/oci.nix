# nix/overlays/container/oci.nix
#
# OCI image extraction and runners
#
{ final }:
let
  inherit (final.aleph) fixed-output-derivation;
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
  # :: { hash : String, name : t1, platform : "linux/amd64", ref : t2 } -> t7
  mk-oci-rootfs =
    {
      name,
      ref,
      hash ? "",
      platform ? "linux/amd64",
    }:
    fixed-output-derivation {
      inherit name hash;
# :: [t6]

      native-build-inputs = [
        final.crane
        final.gnutar
        final.gzip
      # :: String
      ];
# :: String

      SSL_CERT_FILE = "${final.cacert}/etc/ssl/certs/ca-bundle.crt";

      build-script = ''
        # :: { description : "OCI container image rootfs extracted to Nix store" }
        # :: "OCI container image rootfs extracted to Nix store"
        mkdir -p $out
        crane export --platform ${platform} "${ref}" - | tar -xf - -C $out
      '';

      meta = {
        description = "OCI container image rootfs extracted to Nix store";
      };
    };

  # Run OCI images in bwrap/unshare namespace — compiled Haskell, not bash
  inherit (final.aleph.script.compiled) unshare-run;
}
