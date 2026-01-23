#!/usr/bin/env bash
# Buck2 GHC wrapper - filters Mercury-specific flags for stock GHC
filter_args() {
	local skip_next=false
	while IFS= read -r arg || [[ -n "$arg" ]]; do
		if $skip_next; then
			skip_next=false
			continue
		fi
		case "$arg" in
		-dep-json) skip_next=true ;;
		-fpackage-db-byte-code | -fprefer-byte-code | -fbyte-code-and-object-code | -hide-all-packages) ;;
		*) echo "$arg" ;;
		esac
	done
}
final_args=()
for arg in "$@"; do
	if [[ "$arg" == @* ]]; then
		response_file="${arg:1}"
		if [[ -f "$response_file" ]]; then
			filtered_file=$(mktemp)
			filter_args <"$response_file" >"$filtered_file"
			final_args+=("@$filtered_file")
			trap "rm -f '$filtered_file'" EXIT
		else
			final_args+=("$arg")
		fi
	else
		case "$arg" in
		-dep-json | -fpackage-db-byte-code | -fprefer-byte-code | -fbyte-code-and-object-code | -hide-all-packages) ;;
		*) final_args+=("$arg") ;;
		esac
	fi
done
exec ghc "${final_args[@]}"
