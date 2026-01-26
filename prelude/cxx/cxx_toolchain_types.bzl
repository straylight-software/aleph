# prelude/cxx/cxx_toolchain_types.bzl
#
# Minimal extraction from buck2-prelude/cxx/cxx_toolchain_types.bzl (423 lines)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream cxx_toolchain_types.bzl defines:
#   - 4 enums (LinkerType, ShlibInterfacesMode, DepTrackingMode, PicBehavior, CxxObjectFormat)
#   - 10 identical compiler info providers (C, Cxx, Cuda, Hip, Objc, etc.)
#   - LinkerInfo (massive - 40+ fields)
#   - BinaryUtilitiesInfo (nm, objcopy, strip, etc.)
#   - CxxToolchainInfo (the main one)
#   - CxxPlatformInfo
#
# The duplication is absurd:
#   - 10 *CompilerInfo providers with identical fields
#   - LinkerInfo has fields for every platform (darwin, windows, wasm)
#   - Most fields have defaults that are never changed
#
# In Dhall this is ~50 lines with proper type aliases.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

LinkerType = enum("gnu", "darwin", "windows", "wasm")

ShlibInterfacesMode = enum(
    "disabled",
    "enabled",
    "defined_only",
    "stub_from_library",
    "stub_from_headers",
)

PicBehavior = enum(
    "supported",
    "not_supported",
    "always_enabled",
)

CxxObjectFormat = enum(
    "native",
    "bitcode",
    "embedded_bitcode",
)

DepTrackingMode = enum(
    "makefile",
    "show_includes",
    "show_headers",
    "none",
)

# ─────────────────────────────────────────────────────────────────────────────
# Compiler Info (one definition, aliased 10 times upstream)
# ─────────────────────────────────────────────────────────────────────────────

_compiler_fields = {
    "compiler": provider_field(typing.Any, default = None),
    "compiler_type": provider_field(typing.Any, default = None),
    "compiler_flags": provider_field(typing.Any, default = []),
    "preprocessor": provider_field(typing.Any, default = None),
    "preprocessor_type": provider_field(typing.Any, default = None),
    "preprocessor_flags": provider_field(typing.Any, default = []),
    "allow_cache_upload": provider_field(typing.Any, default = None),
    "supports_two_phase_compilation": provider_field(bool, default = False),
}

# All these are identical - upstream repeats this 10 times
CCompilerInfo = provider(fields = _compiler_fields)
CxxCompilerInfo = provider(fields = _compiler_fields)
AsCompilerInfo = provider(fields = _compiler_fields)
AsmCompilerInfo = provider(fields = _compiler_fields)
CudaCompilerInfo = provider(fields = _compiler_fields)
HipCompilerInfo = provider(fields = _compiler_fields)
ObjcCompilerInfo = provider(fields = _compiler_fields)
ObjcxxCompilerInfo = provider(fields = _compiler_fields)
CvtresCompilerInfo = provider(fields = _compiler_fields)
RcCompilerInfo = provider(fields = _compiler_fields)

# ─────────────────────────────────────────────────────────────────────────────
# Linker Info
# ─────────────────────────────────────────────────────────────────────────────

LinkerInfo = provider(
    fields = {
        # Core
        "linker": provider_field(typing.Any, default = None),
        "linker_flags": provider_field(typing.Any, default = []),
        "post_linker_flags": provider_field(typing.Any, default = []),
        "type": provider_field(LinkerType),

        # Archiver
        "archiver": provider_field(typing.Any, default = None),
        "archiver_type": provider_field(typing.Any, default = None),
        "archiver_flags": provider_field(typing.Any, default = None),
        "archiver_supports_argfiles": provider_field(typing.Any, default = None),
        "archiver_reads_inputs": provider_field(bool, default = True),
        "archive_contents": provider_field(typing.Any, default = "normal"),
        "archive_objects_locally": provider_field(typing.Any, default = None),
        "archive_symbol_table": provider_field(bool, default = True),
        "use_archiver_flags": provider_field(typing.Any, default = None),

        # Link behavior
        "link_binaries_locally": provider_field(typing.Any, default = None),
        "link_libraries_locally": provider_field(typing.Any, default = None),
        "link_style": provider_field(typing.Any, default = None),
        "link_weight": provider_field(int, default = 1),
        "link_ordering": provider_field(typing.Any, default = None),

        # LTO
        "lto_mode": provider_field(typing.Any, default = None),
        "supports_distributed_thinlto": provider_field(typing.Any, default = None),
        "dist_thin_lto_codegen_flags": provider_field([cmd_args, None], default = None),
        "thin_lto_premerger_enabled": provider_field(bool, default = False),

        # Shared libraries
        "shlib_interfaces": provider_field(ShlibInterfacesMode),
        "mk_shlib_intf": provider_field(typing.Any, default = None),
        "independent_shlib_interface_linker_flags": provider_field(typing.Any, default = None),
        "shared_library_name_default_prefix": provider_field(typing.Any, default = None),
        "shared_library_name_format": provider_field(typing.Any, default = None),
        "shared_library_versioned_name_format": provider_field(typing.Any, default = None),
        "shared_dep_runtime_ld_flags": provider_field(typing.Any, default = None),

        # Static libraries
        "static_library_extension": provider_field(typing.Any, default = None),
        "static_dep_runtime_ld_flags": provider_field(typing.Any, default = None),
        "static_pic_dep_runtime_ld_flags": provider_field(typing.Any, default = None),

        # File extensions
        "binary_extension": provider_field(typing.Any, default = None),
        "object_file_extension": provider_field(typing.Any, default = None),

        # Misc
        "generate_linker_maps": provider_field(typing.Any, default = None),
        "link_metadata_flag": provider_field(str | None, default = None),
        "sanitizer_runtime_enabled": provider_field(bool, default = False),
        "sanitizer_runtime_files": provider_field(list[Artifact], default = []),
        "requires_archives": provider_field(typing.Any, default = None),
        "requires_objects": provider_field(typing.Any, default = None),
        "force_full_hybrid_if_capable": provider_field(typing.Any, default = None),
        "is_pdb_generated": provider_field(typing.Any, default = None),
        "executable_linker_flags": provider_field(typing.Any, default = []),
        "binary_linker_flags": provider_field(typing.Any, default = []),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Binary Utilities
# ─────────────────────────────────────────────────────────────────────────────

BinaryUtilitiesInfo = provider(fields = {
    "nm": provider_field(typing.Any, default = None),
    "objcopy": provider_field(typing.Any, default = None),
    "objdump": provider_field(typing.Any, default = None),
    "ranlib": provider_field(typing.Any, default = None),
    "strip": provider_field(typing.Any, default = None),
    "dwp": provider_field(typing.Any, default = None),
    "bolt": provider_field(typing.Any, default = None),
    "bolt_msdk": provider_field(typing.Any, default = None),
    "custom_tools": provider_field(dict[str, RunInfo], default = {}),
})

# ─────────────────────────────────────────────────────────────────────────────
# Internal Tools (for header maps, dep files, etc.)
# ─────────────────────────────────────────────────────────────────────────────

CxxInternalTools = provider(fields = {
    "hmap_wrapper": provider_field(typing.Any, default = None),
    "dep_file_processor": provider_field(typing.Any, default = None),
    "dep_file_processor_cmd": provider_field(typing.Any, default = None),
})

# ─────────────────────────────────────────────────────────────────────────────
# Main Toolchain Info
# ─────────────────────────────────────────────────────────────────────────────

CxxToolchainInfo = provider(
    fields = {
        # Compiler infos
        "c_compiler_info": provider_field(typing.Any, default = None),
        "cxx_compiler_info": provider_field(typing.Any, default = None),
        "as_compiler_info": provider_field(typing.Any, default = None),
        "asm_compiler_info": provider_field(typing.Any, default = None),
        "cuda_compiler_info": provider_field(typing.Any, default = None),
        "hip_compiler_info": provider_field(typing.Any, default = None),
        "objc_compiler_info": provider_field(typing.Any, default = None),
        "objcxx_compiler_info": provider_field(typing.Any, default = None),

        # Linker
        "linker_info": provider_field(typing.Any, default = None),
        "binary_utilities_info": provider_field(typing.Any, default = None),

        # Internal tools
        "internal_tools": provider_field(typing.Any, default = None),

        # Headers
        "header_mode": provider_field(typing.Any, default = None),
        "headers_as_raw_headers_mode": provider_field(typing.Any, default = None),
        "raw_headers_as_headers_mode": provider_field(typing.Any, default = None),

        # Compilation
        "cpp_dep_tracking_mode": provider_field(typing.Any, default = None),
        "pic_behavior": provider_field(typing.Any, default = None),
        "object_format": provider_field(typing.Any, default = None),
        "conflicting_header_basename_allowlist": provider_field(typing.Any, default = []),
        "use_dep_files": provider_field(typing.Any, default = None),

        # Linking
        "split_debug_mode": provider_field(typing.Any, default = None),
        "bolt_enabled": provider_field(bool, default = False),
        "llvm_link": provider_field(typing.Any, default = None),

        # Strip
        "strip_flags_info": provider_field(typing.Any, default = None),

        # Dist LTO
        "dist_lto_tools_info": provider_field(typing.Any, default = None),

        # Platform-specific (we don't use these)
        "target_sdk_version": provider_field(typing.Any, default = None),
        "dumpbin_toolchain_path": provider_field(typing.Any, default = None),
    },
)

CxxPlatformInfo = provider(fields = {
    "name": provider_field(typing.Any, default = None),
})
