# nix/overlays/container/firecracker.nix
#
# Firecracker disk image builder
#
{ final, lib }:
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
    in
    run-command name
      {
        native-build-inputs = [
          final.fakeroot
          final.genext2fs
          final.e2fsprogs
        ];
      }
      ''
        mkdir -p $out

        # Copy rootfs to writable location
        cp -a ${rootfs} rootfs
        chmod -R u+w rootfs

        # Inject static busybox for minimal init
        mkdir -p rootfs/usr/local/bin
        cp ${final.pkgsStatic.busybox}/bin/busybox rootfs/usr/local/bin/
        for cmd in sh mount hostname ip cat echo chmod mkdir ln rm ls grep sed tar sleep; do
          ln -sf busybox rootfs/usr/local/bin/$cmd
        done

        ${lib.optionalString (init-script-file != null) ''
          # Install init script (no heredoc)
          install -m755 ${init-script-file} rootfs/init
        ''}

        ${extra-commands}

        # Build ext4 image
        fakeroot genext2fs -B 4096 -b ${to-string size-blocks} -d rootfs $out/disk.ext4
        tune2fs -O extents,uninit_bg,dir_index,has_journal $out/disk.ext4 2>/dev/null || true
      '';
}
