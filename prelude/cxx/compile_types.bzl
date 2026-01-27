# prelude/cxx/compile_types.bzl
#
# Extracted from buck2-prelude

load(":argsfiles.bzl", "CompileArgsfile", "CompileArgsfiles")
load(":cxx_toolchain_types.bzl", "CxxObjectFormat", "DepTrackingMode")

AsmExtensions = enum(
    ".s",
    ".sx",
    ".S",
    ".asm",
    ".asmpp",
)

CxxExtension = enum(
    ".cpp",
    ".cc",
    ".cl",
    ".cxx",
    ".c++",
    ".c",
    ".m",
    ".mm",
    ".cu",
    ".hip",
    ".h",
    ".hpp",
    ".hh",
    ".h++",
    ".hxx",
    ".bc",
    *AsmExtensions.values()
)

HeaderExtension = enum(
    ".h",
    ".hpp",
    ".hh",
    ".h++",
    ".hxx",
    ".cuh",
)

DepFileType = enum(
    "cpp",
    "c",
    "cuda",
    "asm",
)

HeadersDepFiles = record(
    processor = field(cmd_args),
    tag = field(ArtifactTag),
    mk_flags = field(typing.Callable),
    dep_tracking_mode = field(DepTrackingMode),
)

CxxCompileCommand = record(
    base_compile_cmd = field(cmd_args),
    argsfile = field(CompileArgsfile),
    xcode_argsfile = field(CompileArgsfile),
    header_units_argsfile = field(CompileArgsfile | None),
    headers_dep_files = field([HeadersDepFiles, None]),
    compiler_type = field(str),
    category = field(str),
    allow_cache_upload = field(bool),
)

CxxSrcCompileCommand = record(
    src = field(Artifact),
    index = field([int, None], None),
    cxx_compile_cmd = field(CxxCompileCommand),
    args = field(list[typing.Any]),
    is_header = field(bool, False),
    index_store_factory = field(typing.Callable | None, None),
    error_handler = field([typing.Callable, None], None),
)

CxxSrcPrecompileCommand = record(
    src = field(Artifact),
    cxx_compile_cmd = field(CxxCompileCommand),
    args = field(list[typing.Any]),
    extra_argsfile = field([CompileArgsfile, None], None),
)

CxxCompileCommandOutput = record(
    src_compile_cmds = field(list[CxxSrcCompileCommand], default = []),
    base_compile_cmds = field(dict[CxxExtension, CxxCompileCommand], default = {}),
    argsfiles = field(CompileArgsfiles, default = CompileArgsfiles()),
    comp_db_compile_cmds = field(list[CxxSrcCompileCommand], default = []),
)

CxxCompileOutput = record(
    object = field(Artifact),
    object_format = field(CxxObjectFormat, CxxObjectFormat("native")),
    object_has_external_debug_info = field(bool, False),
    external_debug_info = field(Artifact | None, None),
    clang_remarks = field(Artifact | None, None),
    clang_trace = field(Artifact | None, None),
    gcno_file = field(Artifact | None, None),
    index_store = field(Artifact | None, None),
    assembly = field(Artifact | None, None),
    diagnostics = field(Artifact | None, None),
    preproc = field(Artifact | None, None),
    nvcc_dag = field(Artifact | None, None),
    nvcc_env = field(Artifact | None, None),
)

CxxCompileFlavor = enum(
    "default",
    "pic",
    "pic_optimized",
)
