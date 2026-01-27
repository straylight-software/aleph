# prelude/cxx/argsfiles.bzl
#
# Extracted from buck2-prelude

ARGSFILES_SUBTARGET = "argsfiles"

CompileArgsfile = record(
    file = field(Artifact),
    cmd_form = field(cmd_args),
    input_args = field(list[cmd_args]),
    args = field(cmd_args),
    args_without_file_prefix_args = field(cmd_args),
)

CompileArgsfiles = record(
    relative = field(dict[str, CompileArgsfile], default = {}),
    xcode = field(dict[str, CompileArgsfile], default = {}),
)

def get_argsfiles_output(ctx: AnalysisContext, argsfile_by_ext: dict[str, CompileArgsfile], summary_name: str) -> DefaultInfo:
    argsfiles = []
    argsfile_names = []
    dependent_outputs = []
    for _, argsfile in argsfile_by_ext.items():
        argsfiles.append(argsfile.file)
        argsfile_names.append(cmd_args(argsfile.file, ignore_artifacts = True))
        dependent_outputs.extend(argsfile.input_args)

    argsfiles_summary = ctx.actions.write(summary_name, cmd_args(argsfile_names))
    return DefaultInfo(default_outputs = [argsfiles_summary] + argsfiles, other_outputs = dependent_outputs)
