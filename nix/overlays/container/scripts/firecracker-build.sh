mkdir -p $out

# Copy rootfs to writable location
cp -a @rootfs@ rootfs
chmod -R u+w rootfs

# Inject static busybox for minimal init
mkdir -p rootfs/usr/local/bin
cp @busybox@/bin/busybox rootfs/usr/local/bin/
for cmd in sh mount hostname ip cat echo chmod mkdir ln rm ls grep sed tar sleep; do
  ln -sf busybox rootfs/usr/local/bin/$cmd
done

@initScriptInstall@

@extraCommands@

# Build ext4 image
fakeroot genext2fs -B 4096 -b @sizeBlocks@ -d rootfs $out/disk.ext4
tune2fs -O extents,uninit_bg,dir_index,has_journal $out/disk.ext4 2>/dev/null || true
