ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
	ip link set eth0 up 2>/dev/null || true
	ip addr add 172.16.0.2/24 dev eth0 2>/dev/null || true
	ip route add default via 172.16.0.1 2>/dev/null || true
	echo "nameserver 8.8.8.8" >/etc/resolv.conf 2>/dev/null || true
fi
