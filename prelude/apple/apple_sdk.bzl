# prelude/apple/apple_sdk.bzl
#
# Extracted from buck2-prelude

load("@straylight_prelude//apple:apple_toolchain_types.bzl", "AppleToolchainInfo")

def get_apple_sdk_name(ctx: AnalysisContext) -> str:
    """Get the SDK defined on the toolchain."""
    return ctx.attrs._apple_toolchain[AppleToolchainInfo].sdk_name
