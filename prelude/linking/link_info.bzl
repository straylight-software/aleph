# prelude/linking/link_info.bzl
#
# Minimal extraction from buck2-prelude/linking/link_info.bzl (1800+ lines)
# We only need the LinkStyle enum.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream link_info.bzl is massive (1800+ lines) and contains:
#   - LinkStyle enum (what we need)
#   - LinkInfo provider (library linking metadata)
#   - LinkArgs record (link arguments)
#   - MergedLinkInfo provider (combined linking info from deps)
#   - SharedLibraries provider (shared library artifacts)
#   - Complex tset-based link argument propagation
#   - Archive creation helpers
#   - Shared library handling
#
# What's worth keeping (to rewrite in Haskell later):
#   - Transitive link info propagation via tsets
#   - Shared library RPATH handling
#
# What's noise:
#   - 15+ provider types for various linking modes
#   - Complex archive merging logic
#   - Apple framework handling interleaved throughout
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LinkStyle = enum(
    "static",
    "static_pic",
    "shared",
)
