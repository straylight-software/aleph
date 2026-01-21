#!/usr/bin/env bash
# Generate BUCK files from Dhall package definitions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="${SCRIPT_DIR}/../packages-dhall"
TRIPLE="${1:-x86_64-linux-gnu}"

for f in "${PACKAGES_DIR}"/*.dhall; do
	name=$(basename "$f" .dhall)
	echo "# ${name}"
	dhall text <<<"(${SCRIPT_DIR}/Buck.dhall).toStarlark ((${f}) (${SCRIPT_DIR}/Triple.dhall).${TRIPLE})"
	echo
done
