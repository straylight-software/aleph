# nix/modules/flake/container/init-scripts.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // vm init scripts //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Behind the deck, the other construct flickered in and out of focus."
#
#                                                         — Neuromancer
#
# VM init scripts for Isospin (Firecracker) and Cloud Hypervisor.
# Scripts are in ./scripts/ and loaded via builtins.readFile.
#
# TODO: Replace with Nimi (github:weyl-ai/nimi) service definitions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  _class = "flake";

  # Legacy names for compatibility with default.nix
  fc-run-init = builtins.readFile ./scripts/isospin-run-init.sh;
  fc-build-init = builtins.readFile ./scripts/isospin-build-init.sh;
  ch-run-init = builtins.readFile ./scripts/cloud-hypervisor-run-init.sh;
  ch-gpu-init = builtins.readFile ./scripts/cloud-hypervisor-gpu-init.sh;
}
