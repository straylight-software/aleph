#!/usr/bin/env bash
# Generate compile_commands.json for clangd/clang-tidy
set -euo pipefail
TARGETS="${@:-//...}"
echo "Generating compile_commands.json for: $TARGETS"
COMPDB_PATH=$(buck2 bxl prelude//cxx/tools/compilation_database.bxl:generate -- --targets $TARGETS 2>/dev/null | tail -1)
if [ -n "$COMPDB_PATH" ] && [ -f "$COMPDB_PATH" ]; then
	cp "$COMPDB_PATH" compile_commands.json
	echo "Generated compile_commands.json ($(jq length compile_commands.json) entries)"
else
	echo "Failed to generate compile_commands.json" >&2
	exit 1
fi
