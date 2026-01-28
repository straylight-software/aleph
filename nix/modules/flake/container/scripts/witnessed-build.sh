OUTPUT_DIR="$1"
shift
if [ "$1" = "--" ]; then shift; fi

# Clear attestation log before build
ATTESTATION_LOG="/var/log/armitage/fetches.jsonl"
: >"$ATTESTATION_LOG"

# Run the actual command
EXIT_CODE=0
"$@" || EXIT_CODE=$?

# Copy attestations to output directory
if [ -f "$ATTESTATION_LOG" ] && [ -s "$ATTESTATION_LOG" ]; then
  cp "$ATTESTATION_LOG" "$OUTPUT_DIR/.attestations.jsonl"
  FETCH_COUNT=$(wc -l <"$ATTESTATION_LOG")
  echo ":: Witnessed $FETCH_COUNT network fetch(es)"
fi

exit $EXIT_CODE
