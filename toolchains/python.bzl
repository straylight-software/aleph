# nix/build/toolchains/python.bzl
#
# Python toolchain with nanobind for C++ bindings.
#
# Paths are read from .buckconfig.local [python] section.
# Uses Python from Nix devshell with nanobind pre-installed.

def _python_script_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Simple Python script rule.
    """
    return [
        DefaultInfo(default_output = ctx.attrs.main),
        RunInfo(args = [
            read_root_config("python", "interpreter", "python3"),
            ctx.attrs.main,
        ]),
    ]

python_script = rule(
    impl = _python_script_impl,
    attrs = {
        "main": attrs.source(),
        "deps": attrs.list(attrs.dep(), default = []),
    },
)

def _nanobind_extension_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Build a nanobind C++ extension module.

    Uses clang from the :cxx toolchain and nanobind headers from Nix.
    """
    # Get paths from config
    cxx = read_root_config("cxx", "cxx", "clang++")
    python_include = read_root_config("python", "python_include", "/usr/include/python3.12")
    nanobind_include = read_root_config("python", "nanobind_include", "")

    # Output .so file
    out = ctx.actions.declare_output(ctx.attrs.name + ".so")

    # Compile flags
    compile_flags = [
        "-std=c++23",
        "-O2",
        "-fPIC",
        "-shared",
        "-isystem", python_include,
    ]
    if nanobind_include:
        compile_flags.extend(["-isystem", nanobind_include])

    # Build command
    cmd = cmd_args([
        cxx,
    ] + compile_flags + [
        "-o", out.as_output(),
    ] + [src for src in ctx.attrs.srcs])

    ctx.actions.run(cmd, category = "nanobind_compile")

    return [
        DefaultInfo(default_output = out),
    ]

nanobind_extension = rule(
    impl = _nanobind_extension_impl,
    attrs = {
        "srcs": attrs.list(attrs.source()),
        "deps": attrs.list(attrs.dep(), default = []),
        "compiler_flags": attrs.list(attrs.string(), default = []),
    },
)
