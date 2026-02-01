# prelude/apple/apple_toolchain_types.bzl
#
# Extracted from buck2-prelude/apple/apple_toolchain_types.bzl
# No Swift support.

load("@straylight_prelude//cxx:cxx_toolchain_types.bzl", "CxxPlatformInfo", "CxxToolchainInfo")

AppleToolchainInfo = provider(
    fields = {
        "actool": provider_field(RunInfo),
        "architecture": provider_field(str),
        "codesign_allocate": provider_field(RunInfo),
        "codesign_identities_command": provider_field(RunInfo | None, default = None),
        "codesign": provider_field(RunInfo),
        "compile_resources_locally": provider_field(bool),
        "copy_scene_kit_assets": provider_field(RunInfo),
        "cxx_platform_info": provider_field(CxxPlatformInfo),
        "cxx_toolchain_info": provider_field(CxxToolchainInfo),
        "dsymutil": provider_field(RunInfo),
        "dwarfdump": provider_field(RunInfo | None, default = None),
        "extra_linker_outputs": provider_field(list[str]),
        "ibtool": provider_field(RunInfo),
        "installer": provider_field(Label),
        "libtool": provider_field(RunInfo),
        "lipo": provider_field(RunInfo),
        "mapc": provider_field(RunInfo | None, default = None),
        "merge_index_store": provider_field(RunInfo),
        "momc": provider_field(RunInfo),
        "objdump": provider_field(RunInfo | None, default = None),
        "platform_path": provider_field(str | Artifact),
        "sdk_build_version": provider_field(str | None, default = None),
        "sdk_name": provider_field(str),
        "sdk_path": provider_field(str | Artifact),
        "sdk_version": provider_field(str | None, default = None),
        "xcode_build_version": provider_field(str | None, default = None),
        "xcode_version": provider_field(str),
        "xctest": provider_field(RunInfo),
    },
)

AppleToolsInfo = provider(
    fields = {
        "assemble_bundle": provider_field(RunInfo),
        "split_arch_combine_dsym_bundles_tool": provider_field(RunInfo),
        "dry_codesign_tool": provider_field(RunInfo),
        "adhoc_codesign_tool": provider_field(RunInfo),
        "selective_debugging_scrubber": provider_field(RunInfo),
        "info_plist_processor": provider_field(RunInfo),
        "ipa_package_maker": provider_field(RunInfo),
        "make_modulemap": provider_field(RunInfo),
        "make_vfsoverlay": provider_field(RunInfo),
        "framework_sanitizer": provider_field(RunInfo),
        "xcframework_maker": provider_field(RunInfo),
        "static_archive_linker": provider_field(RunInfo),
        "spm_packager": provider_field(RunInfo),
    },
)
