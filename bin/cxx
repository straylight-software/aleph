#!/usr/bin/env bash
set -euo pipefail
get_config() { grep "^${1} = " .buckconfig.local 2>/dev/null | cut -d'=' -f2 | tr -d ' '; }
CXX=$(get_config "cxx")
INCLUDE_ARGS=(
  "-resource-dir" "$(get_config clang_resource_dir)"
  "-isystem" "$(get_config gcc_include)"
  "-isystem" "$(get_config gcc_include_arch)"
  "-isystem" "$(get_config glibc_include)"
)
exec "$CXX" "${INCLUDE_ARGS[@]}" "$@"
