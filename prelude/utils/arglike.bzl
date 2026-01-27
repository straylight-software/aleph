# Command-line argument-like type alias
#
# Extracted from buck2-prelude utils/arglike.bzl
#
# Command-line argument-like. For example, a string, or an artifact.
# Precise list is defined in `ValueAsCommandLineLike::as_command_line`.
# Defining as Any, but can be defined as union type,
# but that might be expensive to check at runtime.
# In the future we will have compiler-time only types,
# and this type could be refined.

ArgLike = typing.Any
