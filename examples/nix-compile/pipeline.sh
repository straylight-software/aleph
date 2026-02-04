#!/usr/bin/env bash
# Pipeline configuration example for nix-compile

# 1. Define pipeline stages using config array
# This pattern allows nix-compile to extract the configuration schema
# even though bash doesn't support complex types natively.
config.pipeline.name="data-ingest"
config.pipeline.retries=3
config.pipeline.timeout=3600

# 2. Define source configuration
config.source.type="s3"
config.source.bucket="${BUCKET:-my-data-bucket}"
config.source.prefix="${PREFIX:-incoming/}"

# 3. Define processing configuration
config.process.batch_size=1000
config.process.compression="zstd"

echo "Starting pipeline: $config.pipeline.name"
echo "Reading from: s3://$config.source.bucket/$config.source.prefix"

# 4. Use whitelisted tools
# mktemp is now allowed!
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# 5. Simulate processing loop
# In a real script, this would loop over files
echo "Processing in $WORK_DIR..."
sleep 1

# 6. Output results
echo "Done."
exit 0
