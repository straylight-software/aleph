# prelude/utils/lazy.bzl
#
# Lazy evaluation utilities.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Starlark's any/all require a list, forcing eager evaluation:
#
#     any([i % 2 == 0 for i in range(100000)])  # Allocates 100k bools
#
# These versions use lazy iteration with zero allocations:
#
#     lazy.is_any(lambda i: i % 2 == 0, range(100000))  # Stops at first True
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_any(predicate, iterable) -> bool:
    """
    Lazy any - returns True if predicate is True for any element.
    Short-circuits on first True, zero allocations.
    """
    for i in iterable:
        if predicate(i):
            return True
    return False

def is_all(predicate, iterable) -> bool:
    """
    Lazy all - returns True if predicate is True for all elements.
    Short-circuits on first False, zero allocations.
    """
    for i in iterable:
        if not predicate(i):
            return False
    return True

def find(predicate, iterable):
    """
    Find first element where predicate is True, or None.
    """
    for i in iterable:
        if predicate(i):
            return i
    return None

def take_while(predicate, iterable) -> list:
    """
    Take elements while predicate is True.
    """
    result = []
    for i in iterable:
        if not predicate(i):
            break
        result.append(i)
    return result

def drop_while(predicate, iterable) -> list:
    """
    Drop elements while predicate is True, return the rest.
    """
    dropping = True
    result = []
    for i in iterable:
        if dropping and predicate(i):
            continue
        dropping = False
        result.append(i)
    return result

lazy = struct(
    is_any = is_any,
    is_all = is_all,
    find = find,
    take_while = take_while,
    drop_while = drop_while,
)
