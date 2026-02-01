# nix/modules/flake/scripts/clang-tidy-check.sh
#
# clang-tidy wrapper for treefmt integration.
# Skips gracefully if compile_commands.json is missing.
#
# Environment variables (substituted by Nix):
#   COMPILE_COMMANDS_PATH - path to compile_commands.json
#   CLANG_TIDY_BIN        - path to clang-tidy binary

if [[ ! -f "$COMPILE_COMMANDS_PATH" ]]; then
	echo "warning: $COMPILE_COMMANDS_PATH not found, skipping clang-tidy" >&2
	exit 0
fi

exec "$CLANG_TIDY_BIN" \
	--warnings-as-errors='*' \
	-p "$COMPILE_COMMANDS_PATH" \
	"$@"
