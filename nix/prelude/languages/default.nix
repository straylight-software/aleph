# nix/prelude/languages/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // languages //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'I'm not Wintermute now.'
#     'So what are you.' He drank from the flask, feeling nothing.
#     'I'm the matrix, Case.'
#
#                                                         — Neuromancer
#
# Language-specific namespaces. The actual namespaces (python, ghc, rust,
# lean) are defined in nix/modules/flake/prelude.nix because they require
# pkgs access. This file exists for structural completeness.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  # Language namespaces are provided by the flake module prelude.nix
}
