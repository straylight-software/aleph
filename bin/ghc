#!/usr/bin/env bash
# Buck2 GHC wrapper
#
# 1. Filters Mercury-specific flags for stock GHC
# 2. Resolves -package X to -package-id <full-id> (GHC 9.12 workaround for
#    Cabal internal library name collisions with ghcWithPackages)
#
# The -package resolution is needed because GHC 9.12 gets confused when
# multiple packages share a name prefix (e.g., vector vs vector-0.13.2.0-*-benchmarks-O2)

# Cache for resolved package ids (avoid repeated ghc-pkg calls)
declare -A PKG_ID_CACHE

resolve_package_id() {
	local pkg="$1"
	if [[ -z "${PKG_ID_CACHE[$pkg]+x}" ]]; then
		# Get the package id from ghc-pkg, strip "id:" prefix and whitespace
		local id
		id=$(ghc-pkg field "$pkg" id 2>/dev/null | sed 's/^id:[[:space:]]*//')
		PKG_ID_CACHE[$pkg]="$id"
	fi
	echo "${PKG_ID_CACHE[$pkg]}"
}

filter_args() {
	local skip_next=false
	local is_package_arg=false
	while IFS= read -r arg || [[ -n "$arg" ]]; do
		if $skip_next; then
			skip_next=false
			continue
		fi
		if $is_package_arg; then
			is_package_arg=false
			local pkg_id
			pkg_id=$(resolve_package_id "$arg")
			if [[ -n "$pkg_id" ]]; then
				echo "-package-id"
				echo "$pkg_id"
			else
				# Fallback if resolution fails
				echo "-package"
				echo "$arg"
			fi
			continue
		fi
		case "$arg" in
		-dep-json) skip_next=true ;;
		-fpackage-db-byte-code | -fprefer-byte-code | -fbyte-code-and-object-code | -hide-all-packages) ;;
		-package) is_package_arg=true ;;
		*) echo "$arg" ;;
		esac
	done
}

# Process command line arguments
# Bash arrays are 0-indexed, but $@ is 1-indexed, so copy to array first
args=("$@")
final_args=()
i=0
while [[ $i -lt ${#args[@]} ]]; do
	arg="${args[$i]}"
	((i++))
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
		-package)
			# Next arg is the package name - resolve to package-id
			if [[ $i -lt ${#args[@]} ]]; then
				pkg_name="${args[$i]}"
				((i++))
				pkg_id=$(resolve_package_id "$pkg_name")
				if [[ -n "$pkg_id" ]]; then
					final_args+=("-package-id" "$pkg_id")
				else
					# Fallback if resolution fails
					final_args+=("-package" "$pkg_name")
				fi
			fi
			;;
		*) final_args+=("$arg") ;;
		esac
	fi
done
exec ghc "${final_args[@]}"
