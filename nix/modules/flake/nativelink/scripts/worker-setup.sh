set -euo pipefail

DATA_DIR="/data"
MARKER="$DATA_DIR/.nix-initialized"

if [ -f "$MARKER" ]; then
	echo "Volume already initialized"
	exit 0
fi

echo "Initializing nix store on volume..."
mkdir -p "$DATA_DIR/nix/store"
mkdir -p "$DATA_DIR/nix/var/nix/db"

# Copy base system from container
echo "Copying base nix store..."
cp -an /nix/store/* "$DATA_DIR/nix/store/" 2>/dev/null || true
cp -an /nix/var/nix/db/* "$DATA_DIR/nix/var/nix/db/" 2>/dev/null || true

touch "$MARKER"
echo "Base nix store initialized. Toolchain will be fetched on demand."
