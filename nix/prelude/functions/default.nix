# nix/prelude/functions/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                          // functional prelude //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Case punched for the Flatline, and watched as Armitage's ice
#     slowly melted, trickle by trickle, into the vast banks of data
#     that made up the construct's memory. Then the grid vanished and
#     the phone began to speak to him.
#
#     It was Dixie, all right, but it wasn't the Flatline's voice.
#     It was the phone itself, the Hosaka's voice synthesis chip
#     reading from RAM.
#
#     'Hey, man,' it said, 'I been trying to reach you.'
#
#                                                         — Neuromancer
#
# The functional prelude. Lists, attrs, strings, maybe, either —
# the building blocks that let you think in functions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
let
  core = import ./core.nix { inherit lib; };
  attrs = import ./attrs.nix { inherit lib; };
  strings = import ./strings.nix { inherit lib; };
  nullable = import ./nullable.nix { inherit lib; };
  logic = import ./logic.nix { inherit lib; };
  types = import ./types.nix { inherit lib; };
  utils = import ./utils.nix { inherit lib; };
  safe = import ./safe.nix { inherit lib; };
in
core
// attrs
// strings
// nullable
// logic
// types
// utils
// {
  # Safe module namespaced to avoid collision with unsafe defaults
  # Use prelude.safe.head instead of prelude.head for total functions
  inherit safe;
}
