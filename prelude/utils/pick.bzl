# Pick utilities - choose between override and underlying values
#
# Extracted from buck2-prelude utils/pick.bzl

def pick(override, underlying):
    """Pick override if not None, else underlying (wrapped in cmd_args)."""
    return cmd_args(override) if override != None else underlying

def pick_bin(override, underlying):
    """Pick override's RunInfo if not None, else underlying."""
    return override[RunInfo] if override != None else underlying

def pick_dep(override, underlying):
    """Pick override if not None, else underlying (raw values)."""
    return pick_raw(override, underlying)

def pick_raw(override, underlying):
    """Pick override if not None, else underlying (raw values)."""
    return override if override != None else underlying

def pick_and_add(override, additional, underlying):
    """Pick override if not None, else underlying, then add additional."""
    flags = [pick(override, underlying)]
    if additional:
        flags.append(additional)
    return cmd_args(flags)
