# Set up networking
ip link set lo up 2>/dev/null || true
if [ -e /sys/class/net/eth0 ]; then
	ip link set eth0 up
	ip addr add 172.16.0.2/24 dev eth0
	ip route add default via 172.16.0.1
	echo "nameserver 8.8.8.8" >/etc/resolv.conf
	echo "nameserver 1.1.1.1" >>/etc/resolv.conf
fi
