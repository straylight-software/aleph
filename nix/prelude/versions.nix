# nix/prelude/versions.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // versions //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'What's he do?'
#     'Same as you, more or less. Professional thief.'
#     'I been outta touch. Who's he?'
#     'Peter Riviera. Sensation artist. Apparently a very good one.
#      Uses holographic implants.'
#
#                                                         — Neuromancer
#
# Pinned language versions. The specific versions of compilers and
# language runtimes that Weyl builds against. Change these when we
# move to a new generation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  python = "3.12";
  ghc = "9.12"; # GHC 9.12 - latest stable before 9.14's doctest/HLS breakage
  lean = "4.15.0";
  rust = "1.92.0";
  clang = "20"; # Required for Blackwell (sm_120)
  gcc = "15";
}
