#!/bin/sh
# Isospin (Firecracker) VM init - interactive shell mode
#
# TODO: Replace with Nimi (github:weyl-ai/nimi) service definition.

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Mount virtual filesystems
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev 2>/dev/null || true
mkdir -p /dev/pts /dev/shm
mount -t devpts devpts /dev/pts 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true
mount -t tmpfs tmpfs /run 2>/dev/null || true
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

# If build command exists, run it and shutdown
if [ -f /build-cmd ]; then
	chmod +x /build-cmd
	/build-cmd
	EXIT=$?
	echo ":: Exit code: $EXIT"
	echo o >/proc/sysrq-trigger
fi

# Interactive mode - show VM info and launch shell
clear
echo "════════════════════════════════════════════════════════"
echo " Isospin VM"
echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
echo "════════════════════════════════════════════════════════"
echo ""

# Launch interactive shell - prefer bash if available
if [ -x /bin/bash ]; then
	exec setsid cttyhack /bin/bash -l
else
	exec setsid cttyhack /bin/sh
fi
