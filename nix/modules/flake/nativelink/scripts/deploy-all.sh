set -euo pipefail

PREFIX="@appPrefix@"
REGION="@region@"
WORKER_COUNT="@workerCount@"
CAS_VOLUME_SIZE="@casVolumeSize@"
WORKER_VOLUME_SIZE="@workerVolumeSize@"
BUILDER_VOLUME_SIZE="@builderVolumeSize@"
WORKER_CPUS="@workerCpus@"
TOTAL_CORES="@totalCores@"
SCHEDULER_FLY_TOML="@schedulerFlyToml@"
CAS_FLY_TOML="@casFlyToml@"
WORKER_FLY_TOML="@workerFlyToml@"
BUILDER_FLY_TOML="@builderFlyToml@"

# Parse args
BUILD_IMAGES=true
for arg in "$@"; do
	case "$arg" in
	--no-build) BUILD_IMAGES=false ;;
	--help | -h)
		echo "Usage: nativelink-deploy [--no-build]"
		echo ""
		echo "Options:"
		echo "  --no-build  Skip container builds, just deploy existing images"
		exit 0
		;;
	esac
done

echo "======================================================================"
echo "           NativeLink Fly.io Deploy"
echo "  ${WORKER_COUNT}x performance-${WORKER_CPUS} workers = ${TOTAL_CORES} cores"
echo "======================================================================"
echo ""

# -- Auth check ---------------------------------------------------------------
echo "Checking authentication..."
if ! gh auth status &>/dev/null; then
	echo "Error: GitHub CLI not authenticated. Run 'gh auth login'"
	exit 1
fi
GH_TOKEN=$(gh auth token)
GH_USER=$(gh api user -q .login)

if ! flyctl auth whoami &>/dev/null; then
	echo "Error: Fly CLI not authenticated. Run 'flyctl auth login'"
	exit 1
fi
echo "  GitHub: $GH_USER"
echo "  Fly.io: $(flyctl auth whoami)"
echo ""

# -- Create apps if needed ----------------------------------------------------
echo "Ensuring Fly apps exist..."
for APP in "$PREFIX-scheduler" "$PREFIX-cas" "$PREFIX-worker"; do
	if ! flyctl apps list --json | jq -e ".[] | select(.Name == \"$APP\")" &>/dev/null; then
		echo "  Creating $APP..."
		flyctl apps create "$APP" --org personal || true
	else
		echo "  $APP exists"
	fi
done
echo ""

# -- Allocate IPs if needed ---------------------------------------------------
echo "Ensuring public IPs allocated..."
for APP in "$PREFIX-scheduler" "$PREFIX-cas"; do
	if ! flyctl ips list -a "$APP" --json | jq -e '.[] | select(.Type == "shared_v4" or .Type == "v4")' &>/dev/null; then
		echo "  Allocating IPv4 for $APP..."
		flyctl ips allocate-v4 --shared -a "$APP"
	fi
	if ! flyctl ips list -a "$APP" --json | jq -e '.[] | select(.Type == "v6")' &>/dev/null; then
		echo "  Allocating IPv6 for $APP..."
		flyctl ips allocate-v6 -a "$APP"
	fi
done
echo "  IPs allocated"
echo ""

# -- Create volumes if needed -------------------------------------------------
echo "Ensuring volumes exist..."
if ! flyctl volumes list -a "$PREFIX-cas" --json | jq -e '.[] | select(.Name == "cas_data")' &>/dev/null; then
	echo "  Creating CAS volume (${CAS_VOLUME_SIZE})..."
	flyctl volumes create cas_data -a "$PREFIX-cas" -r "$REGION" -s "${CAS_VOLUME_SIZE%gb}" -y
fi
for i in $(seq 1 "$WORKER_COUNT"); do
	VOL_NAME="worker_data"
	# Check if we have enough volumes
	VOL_COUNT=$(flyctl volumes list -a "$PREFIX-worker" --json | jq '[.[] | select(.Name == "worker_data")] | length')
	if [ "$VOL_COUNT" -lt "$i" ]; then
		echo "  Creating worker volume $i (${WORKER_VOLUME_SIZE})..."
		flyctl volumes create "$VOL_NAME" -a "$PREFIX-worker" -r "$REGION" -s "${WORKER_VOLUME_SIZE%gb}" -y
	fi
done
echo "  Volumes ready"
echo ""

if [ "$BUILD_IMAGES" = "true" ]; then
	# -- Ensure builder exists --------------------------------------------------
	BUILDER_APP="$PREFIX-builder"
	if ! flyctl apps list --json | jq -e ".[] | select(.Name == \"$BUILDER_APP\")" &>/dev/null; then
		echo "Creating builder app..."
		flyctl apps create "$BUILDER_APP" --org personal || true
	fi

	# Check if builder needs volume
	if ! flyctl volumes list -a "$BUILDER_APP" --json | jq -e '.[] | select(.Name == "builder_nix")' &>/dev/null; then
		echo "Creating builder volume (${BUILDER_VOLUME_SIZE})..."
		flyctl volumes create builder_nix -a "$BUILDER_APP" -r "$REGION" -s "${BUILDER_VOLUME_SIZE%gb}" -y
	fi

	# Check if builder is running
	BUILDER_STATE=$(flyctl status -a "$BUILDER_APP" --json 2>/dev/null | jq -r '.Machines[0].state // "none"' || echo "none")
	if [ "$BUILDER_STATE" != "started" ]; then
		echo "Starting builder..."
		# First time: need to push image from local (bootstrap)
		# After that: builder rebuilds itself
		if ! flyctl status -a "$BUILDER_APP" --json 2>/dev/null | jq -e '.Machines[0]' &>/dev/null; then
			echo "  Bootstrap: building builder image locally (one-time)..."
			nix run ".#nativelink-builder.copyTo" -- \
				--dest-creds "$GH_USER:$GH_TOKEN" \
				"docker://ghcr.io/straylight-software/aleph/nativelink-builder:latest" 2>&1 | tail -2
			FLY_TOKEN=$(flyctl tokens create deploy -a "$BUILDER_APP" -x 2h 2>&1 | head -1)
			skopeo copy \
				--src-creds "$GH_USER:$GH_TOKEN" \
				--dest-creds "x:$FLY_TOKEN" \
				"docker://ghcr.io/straylight-software/aleph/nativelink-builder:latest" \
				"docker://registry.fly.io/$BUILDER_APP:latest" 2>&1 | tail -1
			flyctl deploy -c "$BUILDER_FLY_TOML" -a "$BUILDER_APP" -y 2>&1 | tail -3
		else
			flyctl machines start -a "$BUILDER_APP" "$(flyctl machines list -a "$BUILDER_APP" --json | jq -r '.[0].id')"
		fi
	fi
	echo "  Builder ready"
	echo ""

	# -- Build images on remote builder -----------------------------------------
	echo "Building containers on Fly builder (your laptop stays cool)..."
	REPO_URL="https://github.com/straylight-software/aleph.git"

	for SERVICE in scheduler cas worker; do
		echo "  Building nativelink-$SERVICE..."
		flyctl ssh console -a "$BUILDER_APP" -C "
      set -e
      cd /tmp
      rm -rf aleph 2>/dev/null || true
      git clone --depth 1 $REPO_URL aleph
      cd aleph
      nix run .#nativelink-$SERVICE.copyTo -- \\
        --dest-creds '$GH_USER:$GH_TOKEN' \\
        'docker://ghcr.io/straylight-software/aleph/nativelink-$SERVICE:latest'
    " 2>&1 | tail -5

		echo "  Pushing to Fly registry..."
		FLY_TOKEN=$(flyctl tokens create deploy -a "$PREFIX-$SERVICE" -x 2h 2>&1 | head -1)
		flyctl ssh console -a "$BUILDER_APP" -C "
      skopeo copy \\
        --src-creds '$GH_USER:$GH_TOKEN' \\
        --dest-creds 'x:$FLY_TOKEN' \\
        'docker://ghcr.io/straylight-software/aleph/nativelink-$SERVICE:latest' \\
        'docker://registry.fly.io/$PREFIX-$SERVICE:latest'
    " 2>&1 | tail -2
	done
	echo "  Containers built and pushed"
	echo ""
else
	echo "Skipping container builds (--no-build)"
	echo ""
fi

# -- Deploy services ----------------------------------------------------------
echo "Deploying services..."
flyctl deploy -c "$SCHEDULER_FLY_TOML" -a "$PREFIX-scheduler" -y 2>&1 | tail -2
flyctl deploy -c "$CAS_FLY_TOML" -a "$PREFIX-cas" -y 2>&1 | tail -2
flyctl deploy -c "$WORKER_FLY_TOML" -a "$PREFIX-worker" -y 2>&1 | tail -2
echo "  Services deployed"
echo ""

# -- Scale workers ------------------------------------------------------------
echo "Scaling workers to $WORKER_COUNT..."
flyctl scale count "$WORKER_COUNT" -a "$PREFIX-worker" -y 2>&1 | tail -2
echo "  Workers scaled"
echo ""

# -- Status -------------------------------------------------------------------
echo "======================================================================"
echo "                      Deployment Complete"
echo "======================================================================"
echo ""
echo "Endpoints:"
echo "  Scheduler: grpcs://$PREFIX-scheduler.fly.dev:443"
echo "  CAS:       grpcs://$PREFIX-cas.fly.dev:443"
echo ""
echo "Test with:"
echo "  buck2 build --remote-only \\"
echo "    --config buck2_re_client.engine_address=grpcs://$PREFIX-scheduler.fly.dev:443 \\"
echo "    --config buck2_re_client.cas_address=grpcs://$PREFIX-cas.fly.dev:443 \\"
echo "    //..."
