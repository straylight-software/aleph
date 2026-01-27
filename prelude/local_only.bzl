# prelude/local_only.bzl
#
# Local execution preferences for linking and packaging.
# Controls whether binaries are linked locally vs on remote execution.
#
# Extracted from buck2-prelude/local_only.bzl
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Remote execution (RE) can be slower for linking large binaries.
# This module provides preferences for local vs RE linking.
#
# Key functions:
#   - link_cxx_binary_locally: should this binary link locally?
#   - get_resolved_cxx_binary_link_execution_preference: get preference
#   - package_python_locally: should Python packaging run locally?
#
# Core tools (marked with "is_core_tool" label) always use RE because:
#   - They're small enough
#   - No build stamping, so they cache correctly
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load("@straylight_prelude//cxx:cxx_context.bzl", "get_cxx_toolchain_info")
load(
    "@straylight_prelude//cxx:cxx_toolchain_types.bzl",
    "CxxToolchainInfo",
)

# Link execution preference enum
LinkExecutionPreference = enum(
    "any",         # No preference, can run anywhere
    "local",       # Prefer local execution
    "remote",      # Prefer remote execution
    "full_hybrid", # Use hybrid local/remote
)

def link_cxx_binary_locally(ctx: AnalysisContext, cxx_toolchain: [CxxToolchainInfo, None] = None) -> bool:
    """Determine if a C++ binary should be linked locally."""
    # Core tools are linked on RE because they are
    # a) small enough to do so and
    # b) don't get build stamping so they do cache correctly.
    if _is_core_tool(ctx):
        return False

    return _cxx_toolchain_sets_link_binaries_locally(ctx, cxx_toolchain)

def get_resolved_cxx_binary_link_execution_preference(
        ctx: AnalysisContext,
        links: list[Label],
        force_full_hybrid_if_capable: bool,
        cxx_toolchain: [CxxToolchainInfo, None] = None) -> LinkExecutionPreference:
    """Get the resolved link execution preference for a C++ binary."""
    if force_full_hybrid_if_capable:
        return LinkExecutionPreference("full_hybrid")

    # Core tools can be linked on RE
    if _is_core_tool(ctx):
        return LinkExecutionPreference("any")

    # Check if the toolchain has a preference
    if _cxx_toolchain_sets_link_binaries_locally(ctx, cxx_toolchain):
        return LinkExecutionPreference("local")

    # Default to any
    return LinkExecutionPreference("any")

def _is_core_tool(ctx: AnalysisContext) -> bool:
    """Check if this is a core tool (should use RE)."""
    return "is_core_tool" in getattr(ctx.attrs, "labels", [])

def _cxx_toolchain_sets_link_binaries_locally(ctx: AnalysisContext, cxx_toolchain: [CxxToolchainInfo, None]) -> bool:
    """Check if the toolchain prefers local linking."""
    if not cxx_toolchain:
        cxx_toolchain = get_cxx_toolchain_info(ctx)
    return cxx_toolchain.linker_info.link_binaries_locally
