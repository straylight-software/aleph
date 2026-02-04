#!/usr/bin/env bash
# Deploy script example for nix-compile

# Environment variables with defaults
PORT="${PORT:-8080}"
HOST="${HOST:-localhost}"
API_KEY="${API_KEY:?API_KEY is required}"

# Structured configuration (detected by nix-compile)
config.server.port="$PORT"
config.server.host="$HOST"
config.server.tls=true

# Command usage (curl is whitelisted)
echo "Deploying to $HOST:$PORT..."

if curl --fail --silent "http://$HOST:$PORT/health"; then
	echo "Server is healthy"
else
	echo "Server check failed"
	exit 1
fi

# Example of a store path usage (simulated)
# In a real Nix script, this would be ${pkgs.jq}/bin/jq
# nix-compile will detect this as a bare command if not whitelisted
# jq is NOT whitelisted by default, so this should trigger a warning/error
# unless we use the store path or run in a relaxed mode.
# But for this example, we assume it's provided.
jq -n --arg status "deployed" '{status: $status}'
