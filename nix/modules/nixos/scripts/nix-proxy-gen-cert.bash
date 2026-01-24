#!/usr/bin/env bash
# Generate mitmproxy CA certificate on first boot
set -euo pipefail

cert_dir="@certDir@"
mitmproxy="@mitmproxy@"

if [ ! -f "${cert_dir}/mitmproxy-ca-cert.pem" ]; then
	echo "Generating mitmproxy CA certificate..."
	"${mitmproxy}/bin/mitmdump" --set "confdir=${cert_dir}" -q &
	pid=$!
	sleep 2
	kill "$pid" 2>/dev/null || true
	# Wait for cert to be written
	for _ in $(seq 1 10); do
		[ -f "${cert_dir}/mitmproxy-ca-cert.pem" ] && break
		sleep 0.5
	done
fi
