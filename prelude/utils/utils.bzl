# prelude/utils/utils.bzl
#
# Core utility functions for Starlark.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream utils.bzl contains:
#   - value_or / values_or - null coalescing
#   - flatten / flatten_dict - list/dict flattening
#   - from_named_set - normalize source sets
#   - map_idx / filter_and_map_idx - indexed operations
#   - dedupe_by_value - deduplication
#   - map_val - optional mapping
#
# These are generally useful functional primitives.
#
# The upstream also imports expect.bzl for assertions - we include that here.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def expect(condition: bool, msg: str = "expectation failed"):
    """Assert a condition, failing the build if false."""
    if not condition:
        fail(msg)

def value_or(x, default):
    """Return x if not None, else default."""
    return default if x == None else x

def values_or(*xs):
    """Return first non-None value, or None if all are None."""
    for x in xs:
        if x != None:
            return x
    return None

def flatten(xss: list[list]) -> list:
    """Flatten a list of lists into a single list."""
    return [x for xs in xss for x in xs]

def flatten_dict(xss: list[dict]) -> dict:
    """Flatten a list of dicts into a single dict."""
    return {k: v for xs in xss for k, v in xs.items()}

def map_idx(key, vals: list) -> list:
    """Map index/key access over a list."""
    return [x[key] for x in vals]

def filter_and_map_idx(key, vals: list) -> list:
    """Filter to items with key, then map key access."""
    return [x[key] for x in vals if key in x]

def idx(x, key):
    """Safe index access - returns None if x is None."""
    return x[key] if x != None else None

def dedupe(vals: list) -> list:
    """Remove duplicates from a list, preserving order."""
    seen = {}
    result = []
    for v in vals:
        if v not in seen:
            seen[v] = True
            result.append(v)
    return result

def dedupe_by_value(vals: list) -> list:
    """Remove duplicates using set (order not preserved)."""
    return list({v: True for v in vals}.keys())

def map_val(func, val):
    """Apply func to val if val is not None, else return None."""
    if val == None:
        return None
    return func(val)

def from_named_set(srcs) -> dict:
    """
    Normalize sources to a dict mapping names to artifacts.
    
    If srcs is a list, derive names from short_path.
    If srcs is already a dict, return as-is.
    """
    if type(srcs) == type([]):
        srcs_dict = {}
        for src in srcs:
            if type(src) == "artifact":
                name = src.short_path
            else:
                # Dependency - use default output's short path
                outputs = src[DefaultInfo].default_outputs
                expect(
                    len(outputs) == 1,
                    "expected exactly one default output from {} ({})".format(src, outputs),
                )
                name = outputs[0].short_path
            srcs_dict[name] = src
        return srcs_dict
    else:
        return srcs
