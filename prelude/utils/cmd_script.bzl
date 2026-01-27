# prelude/utils/cmd_script.bzl
#
# Wrap cmd_args as an executable script.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This wraps a cmd_args into a callable script, useful when you need to pass
# an executable + args as a single argument to another tool.
#
# Example (Rust linker wrapper):
#     linker_cmd = cmd_args(linker, linker_flags)
#     wrapper = cmd_script(ctx, "linker", linker_cmd)
#     rustc_args.add("-Clinker={}".format(wrapper))
#
# We only support Unix (sh) since we don't target Windows.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_script(
        ctx: AnalysisContext,
        name: str,
        cmd: cmd_args,
        quote: str | None = "shell") -> cmd_args:
    """
    Wrap cmd_args in a shell script.
    
    Args:
        ctx: analysis context
        name: script name (without .sh extension)
        cmd: command to wrap
        quote: quoting style ("shell" or None)
    
    Returns:
        cmd_args pointing to script with original cmd as hidden dep
    """
    cmd_kwargs = {} if quote == None else {"quote": quote}
    shell_quoted = cmd_args(cmd, **cmd_kwargs)

    wrapper, _ = ctx.actions.write(
        ctx.actions.declare_output("{}.sh".format(name)),
        [
            "#!/usr/bin/env bash",
            cmd_args(cmd_args(shell_quoted, delimiter = " \\\n"), format = "{} \"$@\"\n"),
        ],
        is_executable = True,
        allow_args = True,
    )

    return cmd_args(wrapper, hidden = cmd)
