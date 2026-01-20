# nix/modules/flake/nix-conf.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // nix-conf //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Wintermute. Cold. Cold and silence. A deck with the
#      firmware removed. No personality. Not a person.'
#
#                                                         — Neuromancer
#
# Nix configuration. The Turing registry: all Weyl builds use structured
# attrs. CA derivations temporarily disabled pending nix fork.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  _class = "flake";

  config.flake.nixConfig = {
    # The Turing registry: all Weyl builds use structured attrs
    # (CA derivations temporarily disabled pending nix fork)
  };
}
