set -euo pipefail

SERVICE="@serviceName@"
FLY_APP="@flyApp@"
FLY_CONFIG="@flyConfig@"
GHCR_IMAGE="ghcr.io/straylight-software/aleph/nativelink-@serviceName@:latest"
FLY_IMAGE="registry.fly.io/@flyApp@:latest"

echo "=== Deploying NativeLink $SERVICE ==="

# Get GitHub token for GHCR push
if command -v gh &>/dev/null; then
	GH_TOKEN=$(gh auth token 2>/dev/null || true)
fi
if [ -z "${GH_TOKEN:-}" ]; then
	echo "Error: GitHub CLI not authenticated. Run 'gh auth login' first."
	exit 1
fi

# Get Fly token for registry push
echo "Creating Fly deploy token..."
FLY_TOKEN=$(flyctl tokens create deploy -a "$FLY_APP" -x 2h 2>&1 | head -1)
if [ -z "$FLY_TOKEN" ] || [[ "$FLY_TOKEN" != FlyV1* ]]; then
	echo "Error: Failed to create Fly token. Run 'flyctl auth login' first."
	exit 1
fi

# Build and push to GHCR
echo "Building and pushing to GHCR..."
nix run ".#nativelink-@serviceName@.copyTo" -- \
	--dest-creds "${GITHUB_USER:-$(gh api user -q .login)}:$GH_TOKEN" \
	"docker://$GHCR_IMAGE"

# Copy from GHCR to Fly registry
echo "Copying to Fly registry..."
skopeo copy \
	--src-creds "${GITHUB_USER:-$(gh api user -q .login)}:$GH_TOKEN" \
	--dest-creds "x:$FLY_TOKEN" \
	"docker://$GHCR_IMAGE" \
	"docker://$FLY_IMAGE"

# Deploy to Fly
echo "Deploying to Fly.io..."
flyctl deploy -c "$FLY_CONFIG" -y

echo "=== $SERVICE deployed successfully ==="
flyctl status -a "$FLY_APP"
