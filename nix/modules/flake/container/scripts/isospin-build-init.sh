#!/bin/sh
# Isospin (Firecracker) VM init - build mode
#
# Runs /build-cmd and exits. No interactive shell.
#
# TODO: Replace with Nimi (github:weyl-ai/nimi) service definition.

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Mount virtual filesystems
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true
hostname builder

# Configure network
# Read config from /etc/network-config if present, otherwise use defaults
GUEST_IP="172.16.0.2"
GATEWAY="172.16.0.1"
NETMASK="/30"
if [ -f /etc/network-config ]; then
  # shellcheck source=/dev/null
  . /etc/network-config
fi

ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
  ip link set eth0 up 2>/dev/null || true
  ip addr add "${GUEST_IP}${NETMASK}" dev eth0 2>/dev/null || true
  ip route add default via "${GATEWAY}" 2>/dev/null || true
  echo "nameserver 8.8.8.8" >/etc/resolv.conf
  echo "nameserver 1.1.1.1" >>/etc/resolv.conf
fi

# Run build command if it exists
if [ -f /build-cmd ]; then
  chmod +x /build-cmd
  /build-cmd
  EXIT=$?
  echo ":: Build exit code: $EXIT"
  echo o >/proc/sysrq-trigger
fi

# Fallback to shell if no build command
if [ -x /bin/bash ]; then
  exec /bin/bash
else
  exec /bin/sh
fi
