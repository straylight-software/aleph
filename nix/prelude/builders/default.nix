# nix/prelude/builders/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // builders //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Somewhere, very close, the laugh that wasn't laughter.
#     He never saw Molly again.
#
#                                                         — Neuromancer
#
# Builder functions. The actual builders (fetch, render, script, write)
# are defined in nix/modules/flake/prelude.nix because they require pkgs.
# This file exists for structural completeness in the overlay.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  # Builder namespaces are provided by the flake module prelude.nix
}
