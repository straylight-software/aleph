# nix/modules/flake/scripts/aleph-grep-cpp.sh
#
# ast-grep wrapper for C++ linting with aleph rules.
# Runs each cpp-*.yml rule and exits on first failure.
#
# Environment variables (substituted by Nix):
#   RULES_DIR    - path to linter/rules directory
#   AST_GREP_BIN - path to ast-grep binary

exit_code=0

for rule in "$RULES_DIR"/cpp-*.yml; do
  if [[ -f $rule ]]; then
    "$AST_GREP_BIN" scan --rule "$rule" "$@" || exit_code=1
  fi
done

exit $exit_code
