# Buckconfig reading utilities
#
# Extracted from buck2-prelude utils/buckconfig.bzl
# Stripped: logging features, simplified

load(":expect.bzl", "expect")
load(":lazy.bzl", "lazy")

def _decode_raw_word(val, start, delimiter = None):
    """Read characters up to the given delimiter with quote/escape handling."""
    quotes = ['"', "'"]
    word = ""
    current_quote_char = None
    escaped = False
    idx = -1

    for idx in range(start, len(val)):
        c = val[idx]

        if current_quote_char == None and c == delimiter:
            break

        if current_quote_char == None and c in quotes:
            current_quote_char = c
        elif c == current_quote_char and not escaped:
            current_quote_char = None
        elif c == "\\" and not escaped:
            expect(
                current_quote_char != None,
                "escape char outside of quotes at char %d in: %s" % (idx + 1, val),
            )
            escaped = True
        else:
            word += c
            escaped = False

    expect(current_quote_char == None, "quote not closed in: %s" % val)
    return idx, word

def _next_word(val, start, delimiter):
    """Advance past delimiter characters."""
    for idx in range(start, len(val)):
        c = val[idx]
        if c != delimiter:
            return idx
    return -1

def read(section, field, default = None, root_cell = False):
    """Read a `string` from `.buckconfig`."""
    read_config_func = read_root_config if root_cell else read_config
    return read_config_func(section, field, default)

# Alias for `read` that's explicit about the type being returned.
read_string = read

def read_choice(section, field, choices, default = None, required = True, root_cell = False):
    """Read a string from `.buckconfig` that must be one `choices`."""
    val = read(section, field, root_cell = root_cell)
    if val != None:
        if val in choices:
            return val
        else:
            fail(
                "`{}:{}`: must be one of ({}), but was {}".format(section, field, ", ".join(choices), repr(val)),
            )
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def read_bool(section, field, default = None, required = True, root_cell = False):
    """Read a `boolean` from `.buckconfig`."""
    val = read(section, field, root_cell = root_cell)
    if val != None and val != "":
        if val == "True" or val == "true":
            return True
        elif val == "False" or val == "false":
            return False
        if val.lower() == "true":
            return True
        elif val.lower() == "false":
            return False
        else:
            fail(
                "`{}:{}`: cannot coerce {!r} to bool".format(section, field, val),
            )
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def read_int(section, field, default = None, required = True, root_cell = False):
    """Read an `int` from `.buckconfig`."""
    val = read(section, field, root_cell = root_cell)
    if val != None:
        if val.isdigit():
            return int(val)
        else:
            fail(
                "`{}:{}`: cannot coerce {!r} to int".format(section, field, val),
            )
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def read_list(section, field, delimiter = ",", default = None, required = True, root_cell = False):
    """Read a `list` from `.buckconfig`."""
    val = read(section, field, root_cell = root_cell)
    if val != None:
        quotes = ["\\", '"', "'"]
        if lazy.is_any(lambda x: x in val, quotes):
            words = []
            idx = 0
            for _ in range(len(val)):
                idx = _next_word(val, idx, delimiter)
                if idx == -1:
                    break
                idx, word = _decode_raw_word(val, idx, delimiter)
                words.append(word.strip())
                if idx == -1 or idx >= len(val) - 1:
                    break
            return words
        else:
            return [v.strip() for v in val.split(delimiter) if v]
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def resolve_alias(alias):
    """Resolves an alias into a target (recursively). `fail`s if the alias does not exist."""
    if "//" in alias:
        return alias

    for _ in range(1000):
        target = read("alias", alias, root_cell = False)
        expect(target != None, "Alias {} does not exist".format(alias))
        if "//" in target:
            return target
        else:
            alias = target
    fail("This should never happen - either the alias exists or it doesn't")

# Convenience struct
buckconfig = struct(
    read = read,
    read_string = read_string,
    read_choice = read_choice,
    read_bool = read_bool,
    read_int = read_int,
    read_list = read_list,
    resolve_alias = resolve_alias,
)
