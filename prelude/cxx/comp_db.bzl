# prelude/cxx/comp_db.bzl
#
# Compilation database support for IDE integration.
# Generates compile_commands.json for clangd, ccls, etc.
#
# Extracted from buck2-prelude/cxx/comp_db.bzl (76 lines)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The compilation database is essential for IDE tooling:
#   - clangd uses it for code navigation, completion, diagnostics
#   - ccls, cquery use it similarly
#   - IDE plugins consume compile_commands.json
#
# The upstream implementation:
#   - CxxCompilationDbInfo provider: exposes compilation commands
#   - make_compilation_db_info: creates provider from compile commands
#   - create_compilation_database: generates compile_commands.json
#
# What's worth keeping:
#   - Provider definition (for integration)
#   - Database generation logic (essential for IDEs)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load("@straylight_prelude//:paths.bzl", "paths")
load("@straylight_prelude//cxx:cxx_toolchain_types.bzl", "CxxPlatformInfo", "CxxToolchainInfo")
load("@straylight_prelude//utils:argfile.bzl", "at_argfile")
load(
    "@straylight_prelude//cxx:compile_types.bzl",
    "CxxSrcCompileCommand",
)
load("@straylight_prelude//cxx:cxx_context.bzl", "get_cxx_toolchain_info")

# Provider that exposes the compilation database information.
# Used by IDE tools to understand how to compile each source file.
CxxCompilationDbInfo = provider(fields = {
    # A map of the file (an Artifact) to its corresponding CxxSrcCompileCommand
    "info": provider_field(typing.Any, default = None),
    # Platform for this compilation database
    "platform": provider_field(typing.Any, default = None),
    # Toolchain for this compilation database
    "toolchain": provider_field(typing.Any, default = None),
})

def make_compilation_db_info(
        src_compile_cmds: list[CxxSrcCompileCommand],
        toolchainInfo: CxxToolchainInfo,
        platformInfo: CxxPlatformInfo) -> CxxCompilationDbInfo:
    """Create a CxxCompilationDbInfo from a list of compile commands."""
    info = {}
    for src_compile_cmd in src_compile_cmds:
        info.update({src_compile_cmd.src: src_compile_cmd})

    return CxxCompilationDbInfo(
        info = info,
        toolchain = toolchainInfo,
        platform = platformInfo,
    )

def create_compilation_database(
        ctx: AnalysisContext,
        src_compile_cmds: list[CxxSrcCompileCommand],
        identifier: str) -> DefaultInfo:
    """
    Generate a compile_commands.json for the given source compile commands.
    
    This is the standard format consumed by clangd, ccls, and other tools.
    """
    mk_comp_db = get_cxx_toolchain_info(ctx).internal_tools.make_comp_db

    # Generate the per-source compilation DB entries
    entries = {}
    other_outputs = []

    for src_compile_cmd in src_compile_cmds:
        cdb_path = paths.join(identifier, "__comp_db__", src_compile_cmd.src.short_path + ".comp_db.json")
        if cdb_path not in entries:
            entry = ctx.actions.declare_output(cdb_path)
            cmd = cmd_args(
                mk_comp_db,
                "gen",
                cmd_args(entry.as_output(), format = "--output={}"),
                src_compile_cmd.src.basename,
                cmd_args(src_compile_cmd.src, parent = 1),
                "--",
                src_compile_cmd.cxx_compile_cmd.base_compile_cmd,
                src_compile_cmd.cxx_compile_cmd.argsfile.cmd_form,
                src_compile_cmd.args,
            )
            entry_identifier = paths.join(identifier, src_compile_cmd.src.short_path)
            ctx.actions.run(cmd, category = "cxx_compilation_database", identifier = entry_identifier)

            # Add all inputs the command uses to runtime files
            other_outputs.append(cmd)
            entries[cdb_path] = entry

    # Merge all entries into the actual compilation DB
    db = ctx.actions.declare_output(paths.join(identifier, "compile_commands.json"))
    cmd = cmd_args(mk_comp_db)
    cmd.add("merge")
    cmd.add(cmd_args(db.as_output(), format = "--output={}"))
    cmd.add(at_argfile(
        actions = ctx.actions,
        name = identifier + ".cxx_comp_db_argsfile",
        args = entries.values(),
    ))

    ctx.actions.run(cmd, category = "cxx_compilation_database_merge", identifier = identifier)

    return DefaultInfo(default_output = db, other_outputs = other_outputs)
