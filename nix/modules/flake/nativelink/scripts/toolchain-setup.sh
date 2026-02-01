set -euo pipefail

DATA_DIR="/data"
MARKER="$DATA_DIR/.toolchain-ready"

if [ -f "$MARKER" ]; then
	echo "Toolchain already fetched"
	exit 0
fi

echo "======================================================================"
echo "  Fetching toolchain from cache.nixos.org..."
echo "======================================================================"

# Toolchain store paths (substituted at build time)
TOOLCHAIN_PATHS="@toolchainPaths@"

# Fetch each path from cache
for path in $TOOLCHAIN_PATHS; do
	echo "Fetching $path..."
	nix-store --realise "$path" || echo "  (will build if not in cache)"
done

touch "$MARKER"
echo "Toolchain ready"
