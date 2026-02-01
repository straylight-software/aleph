#!/bin/sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true
hostname builder

ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
	ip link set eth0 up
	ip addr add 172.16.0.2/24 dev eth0
	ip route add default via 172.16.0.1
	echo "nameserver 8.8.8.8" >/etc/resolv.conf
fi

if [ -f /build-cmd ]; then
	chmod +x /build-cmd
	/build-cmd
	EXIT=$?
	echo ":: Build exit code: $EXIT"
	echo o >/proc/sysrq-trigger
fi
# Fall back to sh if bash not available
if [ -x /bin/bash ]; then
	exec /bin/bash
else
	exec /bin/sh
fi
