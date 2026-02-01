# prelude/linking/strip.bzl
#
# Debug symbol stripping utilities.
# Strip debug info from binaries and shared libraries.
#
# Extracted from buck2-prelude/linking/strip.bzl (119 lines)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Stripping is used to:
#   - Reduce binary size for release builds
#   - Create separate debug info files (.debuginfo)
#   - Add .gnu_debuglink for debugger support
#
# Key functions:
#   - strip_debug_info: strip debug info from object
#   - strip_object: strip shared lib/binary
#   - strip_debug_with_gnu_debuglink: split debug info with link
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load("@straylight_prelude//cxx:cxx_context.bzl", "get_cxx_toolchain_info")
load(
    "@straylight_prelude//cxx:cxx_toolchain_types.bzl",
    "CxxToolchainInfo",
    "LinkerType",
)

def _strip_debug_info(ctx: AnalysisContext, out: str, obj: Artifact) -> Artifact:
    """Strip debug information from an object."""
    cxx_toolchain = get_cxx_toolchain_info(ctx)
    strip = cxx_toolchain.binary_utilities_info.strip
    output = ctx.actions.declare_output("__stripped__", out)
    
    if cxx_toolchain.linker_info.type == LinkerType("gnu"):
        cmd = cmd_args([strip, "--strip-debug", "--strip-unneeded", "-o", output.as_output(), obj])
    else:
        # Darwin/other linkers use -S for strip debug
        cmd = cmd_args([strip, "-S", "-o", output.as_output(), obj])
    
    ctx.actions.run(cmd, category = "strip_debug", identifier = out)
    return output

_InterfaceInfo = provider(fields = {
    "artifact": provider_field(typing.Any, default = None),
})

def _anon_strip_debug_info_impl(ctx):
    output = _strip_debug_info(
        ctx = ctx,
        out = ctx.attrs.out,
        obj = ctx.attrs.obj,
    )
    return [DefaultInfo(), _InterfaceInfo(artifact = output)]

# Anonymous wrapper for strip_debug_info
_anon_strip_debug_info = anon_rule(
    impl = _anon_strip_debug_info_impl,
    attrs = {
        "obj": attrs.source(),
        "out": attrs.string(),
        "_cxx_toolchain": attrs.dep(providers = [CxxToolchainInfo]),
    },
    artifact_promise_mappings = {
        "strip_debug_info": lambda p: p[_InterfaceInfo].artifact,
    },
)

def strip_debug_info(
        ctx: AnalysisContext,
        out: str,
        obj: Artifact,
        anonymous: bool = False) -> Artifact:
    """
    Strip debug information from an object file.
    
    Args:
        ctx: Analysis context
        out: Output file name
        obj: Input artifact to strip
        anonymous: Use anonymous target for caching
    """
    if anonymous:
        strip_debug_info_artifact = ctx.actions.anon_target(
            _anon_strip_debug_info,
            dict(
                _cxx_toolchain = ctx.attrs._cxx_toolchain,
                out = out,
                obj = obj,
            ),
        ).artifact("strip_debug_info")
        return ctx.actions.assert_short_path(strip_debug_info_artifact, short_path = out)
    else:
        return _strip_debug_info(ctx = ctx, out = out, obj = obj)

def strip_object(
        ctx: AnalysisContext,
        cxx_toolchain: CxxToolchainInfo,
        unstripped: Artifact,
        strip_flags: cmd_args,
        category_suffix: [str, None] = None,
        output_path: [str, None] = None) -> Artifact:
    """
    Strip unneeded information from binaries / shared libs.
    """
    strip = cxx_toolchain.binary_utilities_info.strip
    output_path = output_path or unstripped.short_path
    stripped_lib = ctx.actions.declare_output("stripped/{}".format(output_path))

    cmd = cmd_args(
        strip,
        strip_flags,
        unstripped,
        "-o",
        stripped_lib.as_output(),
    )

    effective_category_suffix = category_suffix if category_suffix else "shared_lib"
    category = "strip_{}".format(effective_category_suffix)
    ctx.actions.run(cmd, category = category, identifier = output_path)

    return stripped_lib

def strip_debug_with_gnu_debuglink(ctx: AnalysisContext, name: str, obj: Artifact) -> tuple:
    """
    Split a binary into a separate debuginfo binary and a stripped binary
    with a .gnu_debuglink reference.
    
    Returns (stripped_binary, debuginfo_file) tuple.
    """
    objcopy = get_cxx_toolchain_info(ctx).binary_utilities_info.objcopy
    link_locally = get_cxx_toolchain_info(ctx).linker_info.link_binaries_locally

    # Flatten directory structure - .gnu_debuglink doesn't understand directories
    debuginfo_name = name.replace("/", ".")
    debuginfo_output = ctx.actions.declare_output("__debuginfo__", debuginfo_name + ".debuginfo")
    
    cmd = cmd_args([objcopy, "--only-keep-debug", obj, debuginfo_output.as_output()])
    ctx.actions.run(cmd, category = "extract_debuginfo", identifier = name, local_only = link_locally)

    binary_output = ctx.actions.declare_output("__stripped_objects__", name)
    cmd = cmd_args([
        objcopy,
        "--strip-debug",
        "--keep-file-symbols",
        "--add-gnu-debuglink", debuginfo_output,
        obj,
        binary_output.as_output(),
    ])
    ctx.actions.run(cmd, category = "strip_debug", identifier = name, local_only = link_locally)

    return binary_output, debuginfo_output
