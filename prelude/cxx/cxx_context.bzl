# prelude/cxx/cxx_context.bzl
#
# Extracted from buck2-prelude/cxx/cxx_context.bzl
# Apple support, no Swift.

load("@straylight_prelude//apple:apple_toolchain_types.bzl", "AppleToolchainInfo")
load("@straylight_prelude//cxx:cxx_toolchain_types.bzl", "CxxPlatformInfo", "CxxToolchainInfo")

def get_cxx_platform_info(ctx: AnalysisContext) -> CxxPlatformInfo:
    apple_toolchain = getattr(ctx.attrs, "_apple_toolchain", None)
    if apple_toolchain:
        return apple_toolchain[AppleToolchainInfo].cxx_platform_info
    return ctx.attrs._cxx_toolchain[CxxPlatformInfo]

def get_opt_cxx_toolchain_info(ctx: AnalysisContext) -> CxxToolchainInfo | None:
    apple_toolchain = getattr(ctx.attrs, "_apple_toolchain", None)
    if apple_toolchain:
        return apple_toolchain[AppleToolchainInfo].cxx_toolchain_info
    cxx_toolchain = getattr(ctx.attrs, "_cxx_toolchain", None)
    if cxx_toolchain:
        return cxx_toolchain.get(CxxToolchainInfo)
    return None

def get_cxx_toolchain_info(ctx: AnalysisContext) -> CxxToolchainInfo:
    info = get_opt_cxx_toolchain_info(ctx)
    if not info:
        fail("no cxx toolchain info in this ctx")
    return info
