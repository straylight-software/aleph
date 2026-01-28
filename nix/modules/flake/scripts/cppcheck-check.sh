# nix/modules/flake/scripts/cppcheck-check.sh
#
# cppcheck wrapper for treefmt integration.
# Runs deep flow analysis with sensible defaults.
#
# Environment variables (substituted by Nix):
#   CPPCHECK_BIN - path to cppcheck binary

# Suppressed checks:
#   - missingIncludeSystem: system headers are provided by Nix, not visible to cppcheck
#   - unmatchedSuppression: inline suppressions that don't match (different configs)
#   - noExplicitConstructor: style preference, not a bug (allows implicit conversions)
#   - knownConditionTrueFalse: often intentional in tests and assertions
#   - useStlAlgorithm: style preference, raw loops are often clearer
#   - unusedFunction: false positives for __global__ CUDA kernels
#   - dangerousTypeCast: false positives for correct C API casts (zlib, etc)
#   - shiftTooManyBits: false positives for CUDA <<<>>> syntax
#   - constVariablePointer: style, not a bug
#   - unreadVariable: often intentional placeholders
#   - constParameter: style, not a bug
#   - cstyleCast: we use C-style casts when interfacing with C APIs
#   - internalAstError: cppcheck cannot parse CUDA <<<>>> syntax
#   - normalCheckLevelMaxBranches: informational, not an error

exec "$CPPCHECK_BIN" \
  --enable=all \
  --error-exitcode=1 \
  --inline-suppr \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --suppress=noExplicitConstructor \
  --suppress=knownConditionTrueFalse \
  --suppress=useStlAlgorithm \
  --suppress=unusedFunction \
  --suppress=dangerousTypeCast \
  --suppress=shiftTooManyBits \
  --suppress=constVariablePointer \
  --suppress=unreadVariable \
  --suppress=constParameter \
  --suppress=cstyleCast \
  --suppress=internalAstError \
  --suppress=normalCheckLevelMaxBranches \
  --std=c++23 \
  --quiet \
  "$@"
