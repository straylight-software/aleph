# prelude/cxx/headers.bzl
#
# Minimal extraction from buck2-prelude/cxx/headers.bzl (440 lines)
# We only need the HeaderMode enum.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream headers.bzl contains:
#   - HeaderMode enum (what we need)
#   - HeaderStyle enum (local vs system)
#   - CxxHeadersNaming enum (apple vs regular naming)
#   - Headers record (include_path, symlink_tree, etc.)
#   - CHeader record (artifact, name, namespace, named)
#   - Header map generation (_mk_hmap)
#   - Raw headers conversion (as_raw_headers)
#   - Dep file tracking (add_headers_dep_files)
#
# What's worth keeping (to rewrite in Haskell later):
#   - Header maps: .hmap files for fast include resolution
#   - Dep files: track which headers were actually used
#
# What's noise:
#   - Apple-specific naming modes
#   - Complex symlink tree variants
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HeaderMode = enum(
    # Creates the header map that references the headers directly in the source
    # tree.
    "header_map_only",
    # Creates the tree of symbolic links of headers.
    "symlink_tree_only",
    # Creates the tree of symbolic links of headers and creates the header map
    # that references the symbolic links to the headers.
    "symlink_tree_with_header_map",
)
