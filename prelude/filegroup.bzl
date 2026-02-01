# filegroup rule implementation
#
# Extracted from buck2-prelude filegroup.bzl

load("@straylight_prelude//:artifacts.bzl", "ArtifactGroupInfo")

def filegroup_impl(ctx):
    """
    Creates a directory that contains links to the list of srcs.

    The output is a directory that uses `out` for its name, if provided, or the rule name if not.
    Each symlink is based on the `short_path` for the provided `src`.
    """
    output_name = ctx.attrs.out if ctx.attrs.out else ctx.label.name

    if type(ctx.attrs.srcs) == type({}):
        srcs = ctx.attrs.srcs
    else:
        srcs = {}
        for src in ctx.attrs.srcs:
            existing = srcs.get(src.short_path)
            if existing != None and existing != src:
                soft_error(
                    "starlark_filegroup_duplicate_srcs",
                    "filegroup {} has srcs with duplicate names: {} and {}".format(ctx.label, src, srcs[src.short_path]),
                    quiet = True,
                    stack = False,
                )
            srcs[src.short_path] = src

    # buck1 always copies, and that's important for Python rules
    if ctx.attrs.copy:
        output = ctx.actions.copied_dir(output_name, srcs)
    else:
        output = ctx.actions.symlinked_dir(output_name, srcs)

    if type(ctx.attrs.srcs) == type([]):
        artifacts = ctx.attrs.srcs
    else:
        artifacts = [output.project(name, hide_prefix = True) for name in srcs]

    return [
        DefaultInfo(default_output = output),
        ArtifactGroupInfo(artifacts = artifacts),
    ]
