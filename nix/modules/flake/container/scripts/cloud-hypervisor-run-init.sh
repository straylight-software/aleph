#!/bin/sh
# Cloud Hypervisor VM init - interactive shell mode
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
hostname cloud-vm

# Configure network
ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
	ip link set eth0 up
	udhcpc -i eth0 2>/dev/null || (
		ip addr add 172.16.0.2/24 dev eth0
		ip route add default via 172.16.0.1
		echo "nameserver 8.8.8.8" >/etc/resolv.conf
	)
fi

# Interactive mode - show VM info and launch shell
clear
echo "════════════════════════════════════════════════════════"
echo " Cloud Hypervisor VM"
echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
echo "════════════════════════════════════════════════════════"

# Launch interactive shell - prefer bash if available
if [ -x /bin/bash ]; then
	exec setsid cttyhack /bin/bash -l
else
	exec setsid cttyhack /bin/sh
fi
