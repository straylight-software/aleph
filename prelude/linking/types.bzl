# prelude/linking/types.bzl
#
# Extracted from buck2-prelude

# Ways a library can request to be linked (e.g. usually specified via a rule
# param like `preferred_linkage`).
Linkage = enum(
    "any",
    "static",
    "shared",
)
