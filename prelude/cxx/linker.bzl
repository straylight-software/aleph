# prelude/cxx/linker.bzl
#
# Extracted from buck2-prelude

load("@straylight_prelude//cxx:cxx_toolchain_types.bzl", "LinkerInfo", "LinkerType")
load("@straylight_prelude//utils:arglike.bzl", "ArgLike")
load("@straylight_prelude//utils:expect.bzl", "expect")

Linker = record(
    default_shared_library_extension = str,
    default_shared_library_versioned_extension_format = str,
    shared_library_name_linker_flags_format = list[str],
    shared_library_flags = list[str],
)

SharedLibraryFlagOverrides = record(
    shared_library_name_linker_flags_format = list[str],
    shared_library_flags = list[ArgLike],
)

LINKERS = {
    LinkerType("darwin"): Linker(
        default_shared_library_extension = "dylib",
        default_shared_library_versioned_extension_format = "{}.dylib",
        shared_library_name_linker_flags_format = ["-install_name", "@rpath/{}"],
        shared_library_flags = ["-shared"],
    ),
    LinkerType("gnu"): Linker(
        default_shared_library_extension = "so",
        default_shared_library_versioned_extension_format = "so.{}",
        shared_library_name_linker_flags_format = ["-Wl,-soname,{}"],
        shared_library_flags = ["-shared"],
    ),
    LinkerType("wasm"): Linker(
        default_shared_library_extension = "wasm",
        default_shared_library_versioned_extension_format = "{}.wasm",
        shared_library_name_linker_flags_format = [],
        shared_library_flags = ["-shared"],
    ),
    LinkerType("windows"): Linker(
        default_shared_library_extension = "dll",
        default_shared_library_versioned_extension_format = "dll",
        shared_library_name_linker_flags_format = [],
        shared_library_flags = ["/DLL"],
    ),
}

PDB_SUB_TARGET = "pdb"

def _sanitize(s: str) -> str:
    return s.replace("/", "_")

def get_shared_library_name(
        linker_info: LinkerInfo,
        short_name: str,
        apply_default_prefix: bool,
        version: [str, None] = None):
    if version == None:
        full_name = linker_info.shared_library_name_format.format(short_name)
    else:
        full_name = linker_info.shared_library_versioned_name_format.format(short_name, version)
    if apply_default_prefix:
        full_name = linker_info.shared_library_name_default_prefix + full_name
    return full_name

def _parse_ext_macro(name: str) -> [(str, [str, None]), None]:
    if ".$(ext" not in name:
        return None
    expect(name.endswith(")"))
    base, rest = name.split(".$(ext")
    if rest == ")":
        return (base, None)
    expect(rest.startswith(" "))
    return (base, rest[1:-1])

def get_shared_library_name_for_param(linker_info: LinkerInfo, name: str):
    parsed = _parse_ext_macro(name)
    if parsed != None:
        base, version = parsed
        name = get_shared_library_name(
            linker_info,
            base,
            apply_default_prefix = False,
            version = version,
        )
    return name

def get_default_shared_library_name(linker_info: LinkerInfo, label: Label):
    short_name = "{}_{}".format(_sanitize(label.package), _sanitize(label.name))
    return get_shared_library_name(linker_info, short_name, apply_default_prefix = True)

def get_shared_library_name_linker_flags(linker_type: LinkerType, soname: str, flag_overrides: [SharedLibraryFlagOverrides, None] = None) -> list[str]:
    if flag_overrides:
        shared_library_name_linker_flags_format = flag_overrides.shared_library_name_linker_flags_format
    else:
        shared_library_name_linker_flags_format = LINKERS[linker_type].shared_library_name_linker_flags_format
    return [f.format(soname) for f in shared_library_name_linker_flags_format]

def get_shared_library_flags(linker_type: LinkerType, flag_overrides: [SharedLibraryFlagOverrides, None] = None) -> list[ArgLike]:
    if flag_overrides:
        return flag_overrides.shared_library_flags
    return LINKERS[linker_type].shared_library_flags

def get_link_whole_args(linker_type: LinkerType, inputs: list[Artifact]) -> list[typing.Any]:
    args = []
    if linker_type == LinkerType("gnu"):
        args.append("-Wl,--whole-archive")
        args.extend(inputs)
        args.append("-Wl,--no-whole-archive")
    elif linker_type == LinkerType("darwin"):
        for inp in inputs:
            args.append("-Xlinker")
            args.append("-force_load")
            args.append("-Xlinker")
            args.append(inp)
    elif linker_type == LinkerType("windows"):
        for inp in inputs:
            args.append(inp)
            args.append("/WHOLEARCHIVE:" + inp.basename)
    else:
        fail("Linker type {} not supported".format(linker_type))
    return args

def get_objects_as_library_args(linker_type: LinkerType, objects: list[Artifact]) -> list[typing.Any]:
    args = []
    if linker_type == LinkerType("gnu"):
        args.append("-Wl,--start-lib")
        args.extend(objects)
        args.append("-Wl,--end-lib")
    elif linker_type == LinkerType("darwin") or linker_type == LinkerType("windows"):
        args.extend(objects)
    else:
        fail("Linker type {} not supported".format(linker_type))
    return args

def get_ignore_undefined_symbols_flags(linker_type: LinkerType) -> list[str]:
    args = []
    if linker_type == LinkerType("gnu"):
        args.append("-Wl,--allow-shlib-undefined")
        args.append("-Wl,--unresolved-symbols=ignore-all")
    elif linker_type == LinkerType("darwin"):
        args.append("-Wl,-undefined,dynamic_lookup")
    else:
        fail("Linker type {} not supported".format(linker_type))
    return args

def get_no_as_needed_shared_libs_flags(linker_type: LinkerType) -> list[str]:
    args = []
    if linker_type == LinkerType("gnu"):
        args.append("-Wl,--no-as-needed")
    elif linker_type == LinkerType("darwin"):
        pass
    else:
        fail("Linker type {} not supported".format(linker_type))
    return args

def get_output_flags(linker_type: LinkerType, output: Artifact) -> list[ArgLike]:
    if linker_type == LinkerType("windows"):
        return ["/Brepro", cmd_args(output.as_output(), format = "/OUT:{}")]
    else:
        return ["-o", output.as_output()]

def get_import_library(
        ctx: AnalysisContext,
        linker_type: LinkerType,
        output_short_path: str) -> (Artifact | None, list[ArgLike]):
    if linker_type == LinkerType("windows"):
        import_library = ctx.actions.declare_output(output_short_path + ".imp.lib")
        return import_library, [cmd_args(import_library.as_output(), format = "/IMPLIB:{}")]
    else:
        return None, []

def get_deffile_flags(
        ctx: AnalysisContext,
        linker_type: LinkerType) -> list[ArgLike]:
    if linker_type == LinkerType("windows") and ctx.attrs.deffile != None:
        return [cmd_args(ctx.attrs.deffile, format = "/DEF:{}")]
    else:
        return []

def get_rpath_origin(linker_type: LinkerType) -> str:
    if linker_type == LinkerType("gnu"):
        return "$ORIGIN"
    if linker_type == LinkerType("darwin"):
        return "@loader_path"
    fail("Linker type {} not supported".format(linker_type))

def is_pdb_generated(
        linker_type: LinkerType,
        linker_flags: list[[str, ResolvedStringWithMacros]]) -> bool:
    if linker_type != LinkerType("windows"):
        return False
    for flag in reversed(linker_flags):
        flag = str(flag).upper()
        if flag.startswith('"/DEBUG') or flag.startswith('"-DEBUG'):
            return not flag.endswith('DEBUG:NONE"')
    return False

def get_pdb_providers(pdb: Artifact, binary: Artifact):
    return [DefaultInfo(default_output = pdb, other_outputs = [binary])]

DUMPBIN_SUB_TARGET = "dumpbin"

def get_dumpbin_providers(
        ctx: AnalysisContext,
        binary: Artifact,
        dumpbin_toolchain_path: Artifact) -> list[Provider]:
    dumpbin_headers_out = ctx.actions.declare_output(binary.short_path + ".dumpbin_headers")
    ctx.actions.run(
        cmd_args(
            cmd_args(dumpbin_toolchain_path, format = "{}/dumpbin.exe"),
            "/HEADERS",
            binary,
            cmd_args(dumpbin_headers_out.as_output(), format = "/OUT:{}"),
        ),
        category = "dumpbin_headers",
        identifier = binary.short_path,
    )
    return [DefaultInfo(sub_targets = {
        "headers": [DefaultInfo(default_output = dumpbin_headers_out)],
    })]
