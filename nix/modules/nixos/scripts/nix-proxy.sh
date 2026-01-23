#!/usr/bin/env bash
# Wrapper script to start mitmproxy with nix-proxy addon
export NIX_PROXY_CACHE_DIR="@cacheDir@"
export NIX_PROXY_LOG_DIR="@logDir@"
export NIX_PROXY_ALLOWLIST="@allowlist@"

exec @mitmdump@ \
	--listen-host @listenAddress@ \
	--listen-port @port@ \
	--set confdir=@certDir@ \
	--scripts @proxyAddon@ \
	@quietFlag@ \
	"$@"
