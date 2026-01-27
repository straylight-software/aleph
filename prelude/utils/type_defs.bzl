# Type checking utilities
#
# Extracted from buck2-prelude utils/type_defs.bzl

_SELECT_TYPE = type(select({"DEFAULT": []}))

def is_select(thing):
    return type(thing) == _SELECT_TYPE

def is_unicode(arg):
    """Checks if provided instance has a unicode type."""
    return hasattr(arg, "encode")

_STRING_TYPE = type("")

def is_string(arg):
    """Checks if provided instance has a string type."""
    return type(arg) == _STRING_TYPE

_LIST_TYPE = type([])

def is_list(arg):
    """Checks if provided instance has a list type."""
    return type(arg) == _LIST_TYPE

_DICT_TYPE = type({})

def is_dict(arg):
    """Checks if provided instance has a dict type."""
    return type(arg) == _DICT_TYPE

_TUPLE_TYPE = type(())

def is_tuple(arg):
    """Checks if provided instance has a tuple type."""
    return type(arg) == _TUPLE_TYPE

def is_collection(arg):
    """Checks if provided instance is a collection subtype (dict, list, or tuple)."""
    return is_dict(arg) or is_list(arg) or is_tuple(arg)

_BOOL_TYPE = type(True)

def is_bool(arg):
    """Checks if provided instance is a boolean value."""
    return type(arg) == _BOOL_TYPE

_NUMBER_TYPE = type(1)

def is_number(arg):
    """Checks if provided instance is a number value."""
    return type(arg) == _NUMBER_TYPE

_STRUCT_TYPE = type(struct())

def is_struct(arg):
    """Checks if provided instance is a struct value."""
    return type(arg) == _STRUCT_TYPE

def _func():
    pass

_FUNCTION_TYPE = type(_func)

def is_function(arg):
    """Checks if provided instance is a function value."""
    return type(arg) == _FUNCTION_TYPE

type_utils = struct(
    is_bool = is_bool,
    is_number = is_number,
    is_string = is_string,
    is_unicode = is_unicode,
    is_list = is_list,
    is_dict = is_dict,
    is_tuple = is_tuple,
    is_collection = is_collection,
    is_select = is_select,
    is_struct = is_struct,
    is_function = is_function,
)
