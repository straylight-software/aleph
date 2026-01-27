# prelude/cxx/debug.bzl
#
# Extracted from buck2-prelude

# Model the various "split" debug scenarios (e.g. `-gsplit-dwarf`).
SplitDebugMode = enum(
    # Debug info, if present, is inline in the object file.
    "none",
    # Debug info, if present, is in the object file but not linked into binaries.
    "single",
    # Debug info, if present, is separated to .dwo file.
    "split",
)
