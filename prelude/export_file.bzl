# export_file rule implementation
#
# Extracted from buck2-prelude export_file.bzl

def export_file_impl(ctx: AnalysisContext) -> list[DefaultInfo]:
    """Implementation of the export_file build rule."""
    # mode is "copy" or "reference", defaulting to copy
    copy = ctx.attrs.mode != "reference"

    if copy:
        dest = ctx.label.name if ctx.attrs.out == None else ctx.attrs.out
        output = ctx.actions.copy_file(dest, ctx.attrs.src)
    elif ctx.attrs.out != None:
        fail("export_file does not allow specifying `out` without also specifying `mode = 'copy'`")
    else:
        output = ctx.attrs.src
    return [DefaultInfo(default_output = output)]
