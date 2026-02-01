# prelude/utils/dicts.bzl
#
# Dictionary utilities with conflict detection.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The "_x" suffix means "exclusive" - these functions fail on key conflicts
# rather than silently overwriting.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load(":utils.bzl", "expect")

_DEFAULT_FMT = "found different values for key \"{}\": {} != {}"

def update_x(dst: dict, k, v, fmt = _DEFAULT_FMT):
    """
    Set dst[k] = v, failing if dst[k] already exists with a different value.
    """
    p = dst.setdefault(k, v)
    expect(p == v, fmt.format(k, p, v))

def merge_x(dst: dict, src: dict, fmt = _DEFAULT_FMT):
    """
    Merge src into dst, failing on key conflicts.
    """
    for k, v in src.items():
        update_x(dst, k, v, fmt = fmt)

def flatten_x(ds: list[dict], fmt = _DEFAULT_FMT) -> dict:
    """
    Flatten a list of dicts into one, failing on key conflicts.
    """
    out = {}
    for d in ds:
        merge_x(out, d, fmt = fmt)
    return out
