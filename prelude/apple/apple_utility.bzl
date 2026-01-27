# prelude/apple/apple_utility.bzl
#
# Extracted from buck2-prelude
# Swift support stubbed out.

load("@straylight_prelude//apple:apple_toolchain_types.bzl", "AppleToolchainInfo")
load("@straylight_prelude//cxx:headers.bzl", "CxxHeadersLayout", "CxxHeadersNaming")
load("@straylight_prelude//utils:utils.bzl", "value_or")

def get_apple_cxx_headers_layout(ctx: AnalysisContext) -> CxxHeadersLayout:
    namespace = value_or(ctx.attrs.header_path_prefix, ctx.attrs.name)
    return CxxHeadersLayout(namespace = namespace, naming = CxxHeadersNaming("apple"))

def get_module_name(ctx: AnalysisContext) -> str:
    return ctx.attrs.module_name or ctx.attrs.header_path_prefix or ctx.attrs.name

def has_apple_toolchain(ctx: AnalysisContext) -> bool:
    return hasattr(ctx.attrs, "_apple_toolchain")

def get_apple_architecture(ctx: AnalysisContext) -> str:
    return ctx.attrs._apple_toolchain[AppleToolchainInfo].architecture

def get_apple_stripped_attr_value_with_default_fallback(ctx: AnalysisContext) -> bool:
    stripped = ctx.attrs.stripped
    if stripped != None:
        return stripped
    return ctx.attrs._stripped_default

# Swift-specific functions are stubbed - we don't support Swift
def get_disable_pch_validation_flags() -> list[str]:
    """Stubbed - only needed for Swift compilation."""
    return []
