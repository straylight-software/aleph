#!/usr/bin/env bash
# VM init script for Nimi-based inits
set +e # VM init should be robust

# Mount virtual filesystems
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sys /sys 2>/dev/null || true
mount -t devtmpfs dev /dev 2>/dev/null || true
mkdir -p /dev/pts /dev/shm
mount -t devpts devpts /dev/pts 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true
mount -t tmpfs tmpfs /run 2>/dev/null || true

hostname "@hostname@"

@networkSetup@

@gpuSetup@

# Show VM info
clear
echo "════════════════════════════════════════════════════════"
echo " @hostname@"
echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
echo "════════════════════════════════════════════════════════"
echo ""

# Exec into final process
@execInto@
