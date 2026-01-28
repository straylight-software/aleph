# nix/overlays/container/firecracker.nix
#
# Firecracker disk image builder
#
{ final }:
let
  inherit (final.aleph) run-command;
  to-string = builtins.toString;
in
{
  # Build a Firecracker disk image from a rootfs
  #
  # Example:
  #   mk-firecracker-image {
  #     name = "builder-image";
  #     rootfs = mk-oci-rootfs { ... };
  #     size-blocks = 4194304;  # 4GB in 4k blocks
  #     init-script = "#!/bin/sh\nexec /bin/sh";
  #   }
  #
  mk-firecracker-image =
    {
      name,
      rootfs,
      size-blocks ? 4194304, # 4GB in 4k blocks
      init-script ? null,
      extra-commands ? "",
    }:
    let
      init-script-file = if init-script != null then final.writeText "init-script" init-script else null;

      init-script-install =
        if init-script-file != null then "install -m755 ${init-script-file} rootfs/init" else "";
      build-script =
        builtins.replaceStrings
          [ "@rootfs@" "@busybox@" "@initScriptInstall@" "@extraCommands@" "@sizeBlocks@" ]
          [
            (to-string rootfs)
            (to-string final.pkgsStatic.busybox)
            init-script-install
            extra-commands
            (to-string size-blocks)
          ]
          (builtins.readFile ./scripts/firecracker-build.sh);
    in
    run-command name {
      native-build-inputs = [
        final.fakeroot
        final.genext2fs
        final.e2fsprogs
      ];
    } build-script;
}
