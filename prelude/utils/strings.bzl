# prelude/utils/strings.bzl
#
# String utilities.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Simple string manipulation functions that Starlark doesn't provide natively.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strip_prefix(prefix: str, s: str):
    """
    If s starts with prefix, return s with prefix removed.
    Otherwise return None.
    """
    if s.startswith(prefix):
        return s[len(prefix):]
    return None

def strip_suffix(suffix: str, s: str):
    """
    If s ends with suffix, return s with suffix removed.
    Otherwise return None.
    """
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return None

def ensure_prefix(prefix: str, s: str) -> str:
    """Ensure s starts with prefix, adding it if not."""
    if s.startswith(prefix):
        return s
    return prefix + s

def ensure_suffix(suffix: str, s: str) -> str:
    """Ensure s ends with suffix, adding it if not."""
    if s.endswith(suffix):
        return s
    return s + suffix
