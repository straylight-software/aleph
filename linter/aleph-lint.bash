# aleph-lint wrapper script
#
# Runs ast-grep with aleph lint rules.
# Converts absolute paths to relative for proper glob matching.
#
# Environment:
#   SGCONFIG_YML - path to sgconfig.yml
#   AST_GREP_BIN - path to ast-grep binary

# Use unique config file per invocation to avoid race conditions
# when treefmt runs multiple aleph-lint instances in parallel
SGCONFIG_TMP="$(mktemp -t aleph-lint-XXXXXX.yml)"
cp --no-preserve=mode "$SGCONFIG_YML" "$SGCONFIG_TMP"
trap 'rm -f "$SGCONFIG_TMP"' EXIT

# Convert absolute paths to relative paths for glob pattern matching
# treefmt passes absolute paths, but our ignores use relative globs
declare -a REL_ARGS=()
for arg in "$@"; do
	if [[ "$arg" = /* ]] && [[ -f "$arg" ]]; then
		REL_ARGS+=("$(realpath --relative-to=. "$arg")")
	else
		REL_ARGS+=("$arg")
	fi
done

exec "$AST_GREP_BIN" \
	--config "$SGCONFIG_TMP" \
	scan \
	--context 2 \
	--color always \
	"${REL_ARGS[@]}"
