# prelude/linking/link_info.bzl
#
# Extracted from buck2-prelude/linking/link_info.bzl (1013 lines)
# Full extraction for Apple framework support.

load("@straylight_prelude//:artifact_tset.bzl", "ArtifactTSet", "make_artifact_tset")
load("@straylight_prelude//cxx:cxx_toolchain_types.bzl", "LinkerType", "PicBehavior")
load("@straylight_prelude//cxx:linker.bzl", "get_link_whole_args", "get_no_as_needed_shared_libs_flags", "get_objects_as_library_args")
load("@straylight_prelude//linking:types.bzl", "Linkage")
load("@straylight_prelude//utils:arglike.bzl", "ArgLike")

ExtraLinkerOutputs = record(
    artifacts = field(dict[str, Artifact], {}),
    providers = field(dict[str, list[DefaultInfo]], {}),
)

ArchiveContentsType = enum(
    "normal",
    "thin",
    "virtual",
)

Archive = record(
    archive_contents_type = field(ArchiveContentsType, default = ArchiveContentsType("normal")),
    artifact = field(Artifact),
    external_objects = field(list[Artifact], []),
)

LinkStrategy = enum(
    "static",
    "static_pic",
    "shared",
)

LinkStyle = enum(
    "static",
    "static_pic",
    "shared",
)

def to_link_strategy(link_style: LinkStyle) -> LinkStrategy:
    return LinkStrategy(link_style.value)

LibOutputStyle = enum(
    "archive",
    "pic_archive",
    "shared_lib",
)

def default_output_style_for_link_strategy(link_strategy: LinkStrategy) -> LibOutputStyle:
    if link_strategy == LinkStrategy("static"):
        return LibOutputStyle("archive")
    if link_strategy == LinkStrategy("static_pic"):
        return LibOutputStyle("pic_archive")
    return LibOutputStyle("shared_lib")

ArchiveLinkable = record(
    archive = field(Archive),
    bitcode_bundle = field(Artifact | None, None),
    linker_type = field(LinkerType),
    link_whole = field(bool, False),
    supports_lto = field(bool, True),
)

SharedLibLinkable = record(
    lib = field(Artifact),
    link_without_soname = field(bool, False),
)

ObjectsLinkable = record(
    objects = field([list[Artifact], None], None),
    bitcode_bundle = field(Artifact | None, None),
    linker_type = field(LinkerType),
    link_whole = field(bool, False),
)

FrameworksLinkable = record(
    framework_names = field(list[str], []),
    unresolved_framework_paths = field(list[str], []),
    library_names = field(list[str], []),
)

SwiftmoduleLinkable = record(
    swiftmodules = field(ArtifactTSet, ArtifactTSet()),
)

LinkableTypes = [
    ArchiveLinkable,
    SharedLibLinkable,
    ObjectsLinkable,
    FrameworksLinkable,
    SwiftmoduleLinkable,
]

LinkerFlags = record(
    flags = field(list[typing.Any], []),
    post_flags = field(list[typing.Any], []),
    exported_flags = field(list[typing.Any], []),
    exported_post_flags = field(list[typing.Any], []),
)

DepMetadata = record(
    version = field(str),
)

LinkInfo = record(
    name = field([str, None], None),
    dist_thin_lto_codegen_flags = field(list[typing.Any], []),
    pre_flags = field(list[typing.Any], []),
    post_flags = field(list[typing.Any], []),
    linkables = field(list[LinkableTypes], []),
    external_debug_info = field(ArtifactTSet, ArtifactTSet()),
    metadata = field(list[DepMetadata], []),
)

LinkOrdering = enum(
    "preorder",
    "topological",
)

CxxSanitizerRuntimeInfo = provider(fields = {
    "runtime_files": provider_field(list[Artifact]),
})

def set_link_info_link_whole(info: LinkInfo) -> LinkInfo:
    linkables = [set_linkable_link_whole(linkable) for linkable in info.linkables]
    return LinkInfo(
        name = info.name,
        pre_flags = info.pre_flags,
        post_flags = info.post_flags,
        linkables = linkables,
        external_debug_info = info.external_debug_info,
        metadata = info.metadata,
    )

def set_linkable_link_whole(
        linkable: [ArchiveLinkable, ObjectsLinkable, SharedLibLinkable, FrameworksLinkable]) -> [ArchiveLinkable, ObjectsLinkable, SharedLibLinkable, FrameworksLinkable]:
    if isinstance(linkable, ArchiveLinkable):
        return ArchiveLinkable(
            archive = linkable.archive,
            linker_type = linkable.linker_type,
            link_whole = True,
            supports_lto = linkable.supports_lto,
        )
    elif isinstance(linkable, ObjectsLinkable):
        return ObjectsLinkable(
            objects = linkable.objects,
            linker_type = linkable.linker_type,
            link_whole = True,
        )
    return linkable

def wrap_link_info(
        inner: LinkInfo,
        pre_flags: list[typing.Any] = [],
        post_flags: list[typing.Any] = []) -> LinkInfo:
    pre_flags = pre_flags + inner.pre_flags
    post_flags = inner.post_flags + post_flags
    return LinkInfo(
        name = inner.name,
        pre_flags = pre_flags,
        post_flags = post_flags,
        linkables = inner.linkables,
        external_debug_info = inner.external_debug_info,
        metadata = inner.metadata,
    )

def _is_linkable_comprised_of_object_files_or_a_lazy_archive(linkable: LinkableTypes) -> bool:
    return isinstance(linkable, ObjectsLinkable) or (isinstance(linkable, ArchiveLinkable) and not linkable.link_whole)

def append_linkable_args(args: cmd_args, linkable: LinkableTypes):
    if isinstance(linkable, ArchiveLinkable):
        if linkable.archive.archive_contents_type == ArchiveContentsType("virtual"):
            if not linkable.link_whole:
                args.add("-Wl,--start-lib")
            for object in linkable.archive.external_objects:
                args.add(object)
            if not linkable.link_whole:
                args.add("-Wl,--end-lib")
        elif linkable.link_whole:
            args.add(get_link_whole_args(linkable.linker_type, [linkable.archive.artifact]))
        else:
            args.add(linkable.archive.artifact)

        if linkable.archive.archive_contents_type == ArchiveContentsType("thin"):
            args.add(cmd_args(hidden = linkable.archive.external_objects))
    elif isinstance(linkable, SharedLibLinkable):
        if linkable.link_without_soname:
            args.add(cmd_args(linkable.lib, format = "-L{}", parent = 1))
            args.add("-l" + linkable.lib.basename.removeprefix("lib").removesuffix(linkable.lib.extension))
        else:
            args.add(linkable.lib)
    elif isinstance(linkable, ObjectsLinkable):
        if not linkable.link_whole:
            args.add(get_objects_as_library_args(linkable.linker_type, linkable.objects))
        else:
            args.add(linkable.objects)
    elif isinstance(linkable, FrameworksLinkable) or isinstance(linkable, SwiftmoduleLinkable):
        pass
    else:
        fail("Encountered unhandled linkable {}".format(str(linkable)))

LinkInfoArgumentFilter = enum(
    "all",
    "object_files_and_lazy_archives_only",
    "exclude_object_files_and_lazy_archives",
)

def link_info_to_args(value: LinkInfo, argument_type_filter: LinkInfoArgumentFilter = LinkInfoArgumentFilter("all")) -> cmd_args:
    result = cmd_args()
    do_pre_post_flags = argument_type_filter == LinkInfoArgumentFilter("all") or argument_type_filter == LinkInfoArgumentFilter("exclude_object_files_and_lazy_archives")
    if do_pre_post_flags:
        result.add(value.pre_flags)

    for linkable in value.linkables:
        if argument_type_filter == LinkInfoArgumentFilter("all"):
            append_linkable_args(result, linkable)
        elif argument_type_filter == LinkInfoArgumentFilter("object_files_and_lazy_archives_only") and _is_linkable_comprised_of_object_files_or_a_lazy_archive(linkable):
            append_linkable_args(result, linkable)
        elif argument_type_filter == LinkInfoArgumentFilter("exclude_object_files_and_lazy_archives") and not _is_linkable_comprised_of_object_files_or_a_lazy_archive(linkable):
            append_linkable_args(result, linkable)

    if do_pre_post_flags:
        result.add(value.post_flags)
    return result

LinkInfos = record(
    default = field(LinkInfo),
    optimized = field([LinkInfo, None], None),
    stripped = field([LinkInfo, None], None),
)

def _link_info_default_args(infos: LinkInfos):
    return link_info_to_args(infos.default, argument_type_filter = LinkInfoArgumentFilter("all"))

def _link_info_stripped_link_args(infos: LinkInfos):
    info = infos.stripped or infos.default
    return link_info_to_args(info, argument_type_filter = LinkInfoArgumentFilter("all"))

def _link_info_object_files_and_lazy_archives_only_args(infos: LinkInfos):
    return link_info_to_args(infos.default, argument_type_filter = LinkInfoArgumentFilter("object_files_and_lazy_archives_only"))

def _link_info_excluding_object_files_and_lazy_archives_args(infos: LinkInfos):
    return link_info_to_args(infos.default, argument_type_filter = LinkInfoArgumentFilter("exclude_object_files_and_lazy_archives"))

def link_info_to_metadata_args(info: LinkInfo, args: cmd_args | None = None) -> ArgLike:
    if args == None:
        args = cmd_args()
    for meta in info.metadata:
        args.add("version:" + meta.version)
    return args

def _link_info_metadata_args(infos: LinkInfos):
    info = infos.stripped or infos.default
    return link_info_to_metadata_args(info)

LinkInfosTSet = transitive_set(
    args_projections = {
        "default": _link_info_default_args,
        "exclude_object_files_and_lazy_archives": _link_info_excluding_object_files_and_lazy_archives_args,
        "metadata": _link_info_metadata_args,
        "object_files_and_lazy_archives_only": _link_info_object_files_and_lazy_archives_only_args,
        "stripped": _link_info_stripped_link_args,
    },
)

LinkArgsTSet = record(
    infos = field(LinkInfosTSet),
    external_debug_info = field(ArtifactTSet, ArtifactTSet()),
    prefer_stripped = field(bool, False),
)

LinkArgs = record(
    tset = field([LinkArgsTSet, None], None),
    infos = field([list[LinkInfo], None], None),
    flags = field([ArgLike, None], None),
)

LinkedObject = record(
    output = field([Artifact, Promise]),
    bitcode_bundle = field(Artifact | None, None),
    unstripped_output = field(Artifact),
    prebolt_output = field(Artifact | None, None),
    link_args = field(list[LinkArgs] | None, None),
    dwp = field(Artifact | None, None),
    external_debug_info = field(ArtifactTSet, ArtifactTSet()),
    linker_argsfile = field(Artifact | None, None),
    linker_command = field([cmd_args, None], None),
    index_argsfile = field(Artifact | None, None),
    dist_thin_lto_codegen_argsfile = field([Artifact, None], None),
    dist_thin_lto_index_argsfile = field([Artifact, None], None),
    import_library = field(Artifact | None, None),
    pdb = field(Artifact | None, None),
    split_debug_output = field(Artifact | None, None),
)

MergedLinkInfo = provider(fields = [
    "_infos",
    "_external_debug_info",
    "frameworks",
    "swiftmodules",
])

_LIB_OUTPUT_STYLES_FOR_LINKAGE = {
    Linkage("any"): [LibOutputStyle("archive"), LibOutputStyle("pic_archive"), LibOutputStyle("shared_lib")],
    Linkage("static"): [LibOutputStyle("archive"), LibOutputStyle("pic_archive")],
    Linkage("shared"): [LibOutputStyle("pic_archive"), LibOutputStyle("shared_lib")],
}

def wrap_link_infos(
        inner: LinkInfos,
        pre_flags: list[typing.Any] = [],
        post_flags: list[typing.Any] = []) -> LinkInfos:
    return LinkInfos(
        default = wrap_link_info(inner.default, pre_flags = pre_flags, post_flags = post_flags),
        stripped = None if inner.stripped == None else wrap_link_info(inner.stripped, pre_flags = pre_flags, post_flags = post_flags),
    )

def create_merged_link_info(
        ctx: AnalysisContext,
        pic_behavior: PicBehavior,
        link_infos: dict[LibOutputStyle, LinkInfos] = {},
        preferred_linkage: Linkage = Linkage("any"),
        deps: list[MergedLinkInfo] = [],
        exported_deps: list[MergedLinkInfo] = [],
        frameworks_linkable: [FrameworksLinkable, None] = None,
        swiftmodule_linkable: [SwiftmoduleLinkable, None] = None) -> MergedLinkInfo:
    infos = {}
    external_debug_info = {}
    frameworks = {}
    swiftmodules = {}

    for link_strategy in LinkStrategy:
        actual_output_style = get_lib_output_style(link_strategy, preferred_linkage, pic_behavior)
        children = []
        external_debug_info_children = []
        framework_linkables = []
        swiftmodule_linkables = []

        if actual_output_style != LibOutputStyle("shared_lib"):
            framework_linkables.append(frameworks_linkable)
            framework_linkables += [dep_info.frameworks[link_strategy] for dep_info in exported_deps]
            swiftmodule_linkables.append(swiftmodule_linkable)
            swiftmodule_linkables += [dep_info.swiftmodules[link_strategy] for dep_info in exported_deps]

            for dep_info in deps:
                value = dep_info._infos.get(link_strategy)
                if value:
                    children.append(value)
                value = dep_info._external_debug_info.get(link_strategy)
                if value:
                    external_debug_info_children.append(value)
                framework_linkables.append(dep_info.frameworks[link_strategy])
                swiftmodule_linkables.append(dep_info.swiftmodules[link_strategy])

        for dep_info in exported_deps:
            value = dep_info._infos.get(link_strategy)
            if value:
                children.append(value)
            value = dep_info._external_debug_info.get(link_strategy)
            if value:
                external_debug_info_children.append(value)

        frameworks[link_strategy] = merge_framework_linkables(framework_linkables)
        swiftmodules[link_strategy] = merge_swiftmodule_linkables(ctx, swiftmodule_linkables)

        if actual_output_style in link_infos:
            link_info = link_infos[actual_output_style]
            infos[link_strategy] = ctx.actions.tset(LinkInfosTSet, value = link_info, children = children)
            external_debug_info[link_strategy] = make_artifact_tset(
                actions = ctx.actions,
                label = ctx.label,
                children = [link_info.default.external_debug_info] + external_debug_info_children,
            )

    return MergedLinkInfo(
        _infos = infos,
        _external_debug_info = external_debug_info,
        frameworks = frameworks,
        swiftmodules = swiftmodules,
    )

def create_merged_link_info_for_propagation(
        ctx: AnalysisContext,
        xs: list[MergedLinkInfo]) -> MergedLinkInfo:
    merged = {}
    merged_external_debug_info = {}
    frameworks = {}
    swiftmodules = {}
    for link_strategy in LinkStrategy:
        merged[link_strategy] = ctx.actions.tset(
            LinkInfosTSet,
            children = filter(None, [x._infos.get(link_strategy) for x in xs]),
        )
        merged_external_debug_info[link_strategy] = make_artifact_tset(
            actions = ctx.actions,
            label = ctx.label,
            children = filter(None, [x._external_debug_info.get(link_strategy) for x in xs]),
        )
        frameworks[link_strategy] = merge_framework_linkables([x.frameworks[link_strategy] for x in xs])
        swiftmodules[link_strategy] = merge_swiftmodule_linkables(ctx, [x.swiftmodules[link_strategy] for x in xs])

    return MergedLinkInfo(
        _infos = merged,
        _external_debug_info = merged_external_debug_info,
        frameworks = frameworks,
        swiftmodules = swiftmodules,
    )

def get_link_info(infos: LinkInfos, prefer_stripped: bool = False, prefer_optimized: bool = False) -> LinkInfo:
    if prefer_stripped and infos.stripped != None:
        return infos.stripped
    if prefer_optimized and infos.optimized:
        return infos.optimized
    return infos.default

def unpack_link_args(
        args: LinkArgs,
        link_ordering: [LinkOrdering, None] = None,
        link_metadata_flag: str | None = None) -> ArgLike:
    cmd = cmd_args()
    if link_metadata_flag:
        cmd.add(cmd_args(unpack_link_args_metadata(args), prepend = link_metadata_flag))

    if args.tset != None:
        ordering = link_ordering.value if link_ordering else "preorder"
        tset = args.tset.infos
        if args.tset.prefer_stripped:
            cmd.add(tset.project_as_args("stripped", ordering = ordering))
        else:
            cmd.add(tset.project_as_args("default", ordering = ordering))
        return cmd

    if args.infos != None:
        cmd.add([link_info_to_args(info) for info in args.infos])
        return cmd

    if args.flags != None:
        cmd.add(args.flags)
        return cmd

    fail("Unpacked invalid empty link args")

def unpack_link_args_metadata(args: LinkArgs) -> ArgLike:
    if args.tset != None:
        return args.tset.infos.project_as_args("metadata")
    ret = cmd_args()
    if args.infos != None:
        for info in args.infos:
            link_info_to_metadata_args(info, ret)
    return ret

def unpack_external_debug_info(actions: AnalysisActions, args: LinkArgs) -> ArtifactTSet:
    if args.tset != None:
        if args.tset.prefer_stripped:
            return ArtifactTSet()
        return args.tset.external_debug_info
    if args.infos != None:
        return make_artifact_tset(actions = actions, children = [info.external_debug_info for info in args.infos])
    if args.flags != None:
        return ArtifactTSet()
    fail("Unpacked invalid empty link args")

def get_link_args_for_strategy(
        ctx: AnalysisContext,
        deps_merged_link_infos: list[MergedLinkInfo],
        link_strategy: LinkStrategy,
        prefer_stripped: bool = False,
        additional_link_info: [LinkInfo, None] = None) -> LinkArgs:
    infos_kwargs = {}
    if additional_link_info:
        infos_kwargs = {"value": LinkInfos(default = additional_link_info, stripped = additional_link_info)}
    infos = ctx.actions.tset(
        LinkInfosTSet,
        children = filter(None, [x._infos.get(link_strategy) for x in deps_merged_link_infos]),
        **infos_kwargs
    )
    external_debug_info = make_artifact_tset(
        actions = ctx.actions,
        label = ctx.label,
        children = filter(
            None,
            [x._external_debug_info.get(link_strategy) for x in deps_merged_link_infos] + ([additional_link_info.external_debug_info] if additional_link_info else []),
        ),
    )
    return LinkArgs(
        tset = LinkArgsTSet(
            infos = infos,
            external_debug_info = external_debug_info,
            prefer_stripped = prefer_stripped,
        ),
    )

def get_lib_output_style(
        requested_link_strategy: LinkStrategy,
        preferred_linkage: Linkage,
        pic_behavior: PicBehavior) -> LibOutputStyle:
    no_pic_style = _get_lib_output_style_without_pic_behavior(requested_link_strategy, preferred_linkage)
    return process_output_style_for_pic_behavior(no_pic_style, pic_behavior)

def _get_lib_output_style_without_pic_behavior(requested_link_strategy: LinkStrategy, preferred_linkage: Linkage) -> LibOutputStyle:
    if preferred_linkage == Linkage("any"):
        return default_output_style_for_link_strategy(requested_link_strategy)
    elif preferred_linkage == Linkage("shared"):
        return LibOutputStyle("shared_lib")
    else:
        if requested_link_strategy == LinkStrategy("static"):
            return LibOutputStyle("archive")
        else:
            return LibOutputStyle("pic_archive")

def process_link_strategy_for_pic_behavior(link_strategy: LinkStrategy, behavior: PicBehavior) -> LinkStrategy:
    if link_strategy == LinkStrategy("shared"):
        return link_strategy
    elif behavior == PicBehavior("supported"):
        return link_strategy
    elif behavior == PicBehavior("not_supported"):
        return LinkStrategy("static")
    elif behavior == PicBehavior("always_enabled"):
        return LinkStrategy("static_pic")
    else:
        fail("Unknown pic_behavior: {}".format(behavior))

def process_output_style_for_pic_behavior(output_style: LibOutputStyle, behavior: PicBehavior) -> LibOutputStyle:
    if output_style == LibOutputStyle("shared_lib"):
        return output_style
    elif behavior == PicBehavior("supported"):
        return output_style
    elif behavior == PicBehavior("not_supported"):
        return LibOutputStyle("archive")
    elif behavior == PicBehavior("always_enabled"):
        return LibOutputStyle("pic_archive")
    else:
        fail("Unknown pic_behavior: {}".format(behavior))

def subtarget_for_output_style(output_style: LibOutputStyle) -> str:
    return legacy_output_style_to_link_style(output_style).value.replace("_", "-")

def get_output_styles_for_linkage(linkage: Linkage) -> list[LibOutputStyle]:
    return _LIB_OUTPUT_STYLES_FOR_LINKAGE[linkage]

def legacy_output_style_to_link_style(output_style: LibOutputStyle) -> LinkStyle:
    if output_style == LibOutputStyle("shared_lib"):
        return LinkStyle("shared")
    elif output_style == LibOutputStyle("archive"):
        return LinkStyle("static")
    elif output_style == LibOutputStyle("pic_archive"):
        return LinkStyle("static_pic")
    fail("unrecognized output_style {}".format(output_style))

def merge_framework_linkables(linkables: list[[FrameworksLinkable, None]]) -> FrameworksLinkable:
    unique_framework_names = {}
    unique_framework_paths = {}
    unique_library_names = {}
    for linkable in linkables:
        if not linkable:
            continue
        for framework in linkable.framework_names:
            unique_framework_names[framework] = True
        for framework_path in linkable.unresolved_framework_paths:
            unique_framework_paths[framework_path] = True
        for library_name in linkable.library_names:
            unique_library_names[library_name] = True

    return FrameworksLinkable(
        framework_names = unique_framework_names.keys(),
        unresolved_framework_paths = unique_framework_paths.keys(),
        library_names = unique_library_names.keys(),
    )

def merge_swiftmodule_linkables(ctx: AnalysisContext, linkables: list[[SwiftmoduleLinkable, None]]) -> SwiftmoduleLinkable:
    return SwiftmoduleLinkable(swiftmodules = make_artifact_tset(
        actions = ctx.actions,
        label = ctx.label,
        children = [linkable.swiftmodules for linkable in linkables if linkable != None],
    ))

def wrap_with_no_as_needed_shared_libs_flags(linker_type: LinkerType, link_info: LinkInfo) -> LinkInfo:
    if linker_type == LinkerType("gnu"):
        return wrap_link_info(
            inner = link_info,
            pre_flags = ["-Wl,--push-state"] + get_no_as_needed_shared_libs_flags(linker_type),
            post_flags = ["-Wl,--pop-state"],
        )
    if linker_type == LinkerType("darwin"):
        return link_info
    fail("Linker type {} not supported".format(linker_type))

LinkCommandDebugOutput = record(
    filename = str,
    command = ArgLike,
    argsfile = Artifact,
    dist_thin_lto_codegen_argsfile = Artifact | None,
    dist_thin_lto_index_argsfile = Artifact | None,
)

LinkCommandDebugOutputInfo = provider(fields = ["debug_outputs"])

UnstrippedLinkOutputInfo = provider(fields = {"artifact": Artifact})

def make_link_command_debug_output(linked_object: LinkedObject) -> [LinkCommandDebugOutput, None]:
    local_link_debug_info_present = linked_object.output and linked_object.linker_command and linked_object.linker_argsfile
    distributed_link_debug_info_present = linked_object.dist_thin_lto_index_argsfile and linked_object.dist_thin_lto_codegen_argsfile
    if not local_link_debug_info_present and not distributed_link_debug_info_present:
        return None
    return LinkCommandDebugOutput(
        filename = linked_object.output.short_path,
        command = linked_object.linker_command,
        argsfile = linked_object.linker_argsfile,
        dist_thin_lto_index_argsfile = linked_object.dist_thin_lto_index_argsfile,
        dist_thin_lto_codegen_argsfile = linked_object.dist_thin_lto_codegen_argsfile,
    )

def make_link_command_debug_output_json_info(ctx: AnalysisContext, debug_outputs: list[LinkCommandDebugOutput]) -> Artifact:
    json_info = []
    associated_artifacts = []
    for debug_output in debug_outputs:
        is_distributed_link = debug_output.dist_thin_lto_index_argsfile and debug_output.dist_thin_lto_codegen_argsfile
        if is_distributed_link:
            json_info.append({
                "dist_thin_lto_codegen_argsfile": debug_output.dist_thin_lto_codegen_argsfile,
                "dist_thin_lto_index_argsfile": debug_output.dist_thin_lto_index_argsfile,
                "filename": debug_output.filename,
            })
            associated_artifacts.extend([debug_output.dist_thin_lto_codegen_argsfile, debug_output.dist_thin_lto_index_argsfile])
        else:
            json_info.append({
                "argsfile": debug_output.argsfile,
                "command": debug_output.command,
                "filename": debug_output.filename,
            })
            associated_artifacts.extend(filter(None, [debug_output.argsfile]))

    json_output = ctx.actions.write_json("linker.command", json_info, with_inputs = False)
    json_output_with_artifacts = json_output.with_associated_artifacts(associated_artifacts)
    return json_output_with_artifacts
