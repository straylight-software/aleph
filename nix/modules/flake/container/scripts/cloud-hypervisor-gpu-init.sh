#!/bin/sh
# Cloud Hypervisor VM init - GPU passthrough mode
#
# Waits for GPU device, loads modules, then launches shell.
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
hostname ch-gpu

# Configure network
ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
	ip link set eth0 up 2>/dev/null || true
	ip addr add 172.16.0.2/24 dev eth0 2>/dev/null || true
	ip route add default via 172.16.0.1 2>/dev/null || true
	echo "nameserver 8.8.8.8" >/etc/resolv.conf
fi

# Wait for GPU device
echo ":: Waiting for GPU..."
for i in $(seq 1 30); do
	if [ -e /dev/nvidia0 ]; then
		echo ":: GPU ready"
		break
	fi
	sleep 0.1
done

# Load NVIDIA modules
modprobe nvidia 2>/dev/null || true
modprobe nvidia_uvm 2>/dev/null || true

# Run build command if it exists
if [ -f /build-cmd ]; then
	chmod +x /build-cmd
	echo ":: Running command"
	/build-cmd
	EXIT=$?
	echo ":: Exit code: $EXIT"
	echo o >/proc/sysrq-trigger
fi

# Launch interactive shell - prefer bash if available
if [ -x /bin/bash ]; then
	exec setsid cttyhack /bin/bash -l
else
	exec setsid cttyhack /bin/sh
fi
