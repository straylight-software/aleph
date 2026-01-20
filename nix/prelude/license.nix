# nix/prelude/license.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                               // license //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Nobody trusts those fuckers, you know that. Every AI ever built
#      has an electromagnetic shotgun wired to its forehead.'
#
#                                                         — Neuromancer
#
# License aliases. Human-readable names for the legalese.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
{
  inherit (lib.licenses) mit;
  asl-2-0 = lib.licenses.asl20;
  bsd-2 = lib.licenses.bsd2;
  bsd-3 = lib.licenses.bsd3;
  gpl-2 = lib.licenses.gpl2Only;
  gpl-3 = lib.licenses.gpl3Only;
  lgpl-2 = lib.licenses.lgpl2Only;
  lgpl-3 = lib.licenses.lgpl3Only;
  mpl-2-0 = lib.licenses.mpl20;
  apache = lib.licenses.asl20;
  inherit (lib.licenses) unfree;
}
