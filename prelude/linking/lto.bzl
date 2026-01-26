# prelude/linking/lto.bzl
#
# Minimal extraction from buck2-prelude/linking/lto.bzl (~80 lines)
# We only need the LtoMode enum.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream lto.bzl contains:
#   - LtoMode enum (what we need)
#   - lto_compiler_flags() - returns -flto flags for mode
#   - lto_linker_flags() - returns linker flags for mode
#
# What's worth keeping:
#   - LTO mode selection is useful
#   - The flag generation is trivial (just -flto=thin etc)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LtoMode = enum(
    # No LTO
    "none",
    # Object files contain both LTO IR and native code to allow binaries to link
    # either via standard or LTO.
    "fat",
    # Traditional, monolithic LTO.
    "monolithic",
    # https://clang.llvm.org/docs/ThinLTO.html
    "thin",
)
