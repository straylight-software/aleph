# nix/modules/flake/scripts/cppcheck-check.sh
#
# cppcheck wrapper for treefmt integration.
# Runs deep flow analysis with sensible defaults.
#
# Environment variables (substituted by Nix):
#   CPPCHECK_BIN - path to cppcheck binary

exec "$CPPCHECK_BIN" \
	--enable=all \
	--error-exitcode=1 \
	--inline-suppr \
	--suppress=missingIncludeSystem \
	--suppress=unmatchedSuppression \
	--std=c++23 \
	--quiet \
	"$@"
