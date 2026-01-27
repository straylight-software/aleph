# prelude/utils/argfile.bzl
#
# Argument file utilities for long command lines.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# When command lines get too long (common with many include paths or libs),
# we write args to a file and pass @argfile to the tool.
#
# Two variants:
#   at_argfile - returns "@path/to/file" (for tools that expect @ prefix)
#   argfile    - returns just the file path (for manual formatting)
#
# Both return cmd_args with the original args as hidden deps.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def at_argfile(
        *,
        actions,
        name,
        args,
        allow_args: bool = False) -> cmd_args:
    """
    Write args to a file and return @path/to/argfile.
    
    Args:
        actions: ctx.actions
        name: argument file name
        args: arguments to write
        allow_args: allow cmd_args in args
    
    Returns:
        cmd_args with @filename format and args as hidden deps
    """
    if allow_args:
        args_file, _ = actions.write(name, args, allow_args = True, with_inputs = True)
    else:
        args_file = actions.write(name, args, with_inputs = True)
    return cmd_args(args_file, format = "@{}", hidden = args)

def argfile(
        *,
        actions,
        name,
        args,
        allow_args: bool = False) -> cmd_args:
    """
    Write args to a file and return the file path.
    
    Args:
        actions: ctx.actions
        name: argument file name
        args: arguments to write
        allow_args: allow cmd_args in args
    
    Returns:
        cmd_args with filename and args as hidden deps
    """
    if allow_args:
        args_file, _ = actions.write(name, args, allow_args = True, with_inputs = True)
    else:
        args_file = actions.write(name, args, with_inputs = True)
    return cmd_args(args_file, hidden = args)
