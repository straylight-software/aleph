# prelude/paths.bzl
#
# Path manipulation utilities (from Bazel Skylib).
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This is the Bazel Skylib paths module, adapted for Buck2.
# Unix-style paths only (forward slashes).
#
# Functions:
#   basename     - file portion of path
#   dirname      - directory portion of path
#   is_absolute  - check if path is absolute
#   join         - join path components
#   normalize    - normalize path (remove . and ..)
#   relativize   - make path relative to another
#   replace_extension - change file extension
#   split_extension   - split path into (root, ext)
#   starts_with  - check if path starts with prefix
#   strip_suffix - remove suffix from path
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def basename(p: str) -> str:
    """Returns the basename (file portion) of a path."""
    return p.rpartition("/")[-1]

def dirname(p: str) -> str:
    """Returns the dirname (directory portion) of a path."""
    prefix, sep, _ = p.rpartition("/")
    if not prefix:
        return sep
    return prefix.rstrip("/")

def is_absolute(path: str) -> bool:
    """Returns True if path is absolute."""
    return path.startswith("/")

def join(path: str, *others) -> str:
    """Join path components, like os.path.join."""
    result = path
    for p in others:
        if is_absolute(p):
            result = p
        elif not result or result.endswith("/"):
            result += p
        else:
            result += "/" + p
    return result

def normalize(path: str) -> str:
    """Normalize path, eliminating . and .. segments."""
    if not path:
        return "."

    if path.startswith("//") and not path.startswith("///"):
        initial_slashes = 2
    elif path.startswith("/"):
        initial_slashes = 1
    else:
        initial_slashes = 0
    is_relative = (initial_slashes == 0)

    components = path.split("/")
    new_components = []

    for component in components:
        if component in ("", "."):
            continue
        if component == "..":
            if new_components and new_components[-1] != "..":
                new_components.pop()
            elif is_relative:
                new_components.append(component)
        else:
            new_components.append(component)

    path = "/".join(new_components)
    if not is_relative:
        path = ("/" * initial_slashes) + path

    return path or "."

def relativize(path: str, start: str) -> str:
    """Returns portion of path relative to start."""
    if path == start:
        return ""
    segments = normalize(path).split("/")
    start_segments = normalize(start).split("/")
    if start_segments == ["."]:
        start_segments = []
    start_length = len(start_segments)

    if path.startswith("..") or start.startswith(".."):
        fail("Cannot relativize paths above the current directory")

    if (path.startswith("/") != start.startswith("/") or
        len(segments) < start_length):
        fail("Path '{}' is not beneath '{}'".format(path, start))

    for ancestor_segment, segment in zip(start_segments, segments):
        if ancestor_segment != segment:
            fail("Path '{}' is not beneath '{}'".format(path, start))

    length = len(segments) - start_length
    result_segments = segments[-length:] if length > 0 else []
    return "/".join(result_segments)

def replace_extension(p: str, new_extension: str) -> str:
    """Replace file extension."""
    root, _ = split_extension(p)
    return root + new_extension

def split_extension(p: str) -> (str, str):
    """Split path into (root, extension)."""
    b = basename(p)
    last_dot = b.rfind(".")
    if last_dot <= 0:
        return (p, "")
    dot_distance = len(b) - last_dot
    return (p[:-dot_distance], p[-dot_distance:])

def starts_with(path: str, prefix: str) -> bool:
    """Check if path starts with prefix."""
    return path == prefix or path.startswith(prefix + "/")

def strip_suffix(a: str, b: str):
    """Strip suffix b from path a, or return None."""
    if len(b) > len(a):
        return None
    pa = a.split("/")
    pb = b.split("/")
    if len(pb) > len(pa):
        return None
    for idx in range(len(pb)):
        if pb[len(pb) - 1 - idx] != pa[len(pa) - 1 - idx]:
            return None
    return "/".join(pa[:len(pa) - len(pb)])

# Struct for namespaced access
paths = struct(
    basename = basename,
    dirname = dirname,
    is_absolute = is_absolute,
    join = join,
    normalize = normalize,
    relativize = relativize,
    replace_extension = replace_extension,
    split_extension = split_extension,
    starts_with = starts_with,
    strip_suffix = strip_suffix,
)
