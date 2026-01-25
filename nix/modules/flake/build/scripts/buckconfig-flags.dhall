-- nix/modules/flake/build/scripts/buckconfig-flags.dhall
--
-- Turing Registry flags for Buck2 toolchains
-- The true names of Straylight derivations.
--
-- These flags come from nix/prelude/turing-registry.nix and are
-- non-negotiable. Code that builds under these flags can be debugged.

let c_flags : Text = env:C_FLAGS as Text
let cxx_flags : Text = env:CXX_FLAGS as Text

in ''
# ════════════════════════════════════════════════════════════════════════════════
# Turing Registry - Non-negotiable build flags
# ════════════════════════════════════════════════════════════════════════════════
# Source: nix/prelude/turing-registry.nix
# These flags ensure all code can be debugged. Do not override.

[cxx.flags]
c_flags = ${c_flags}
cxx_flags = ${cxx_flags}
''
