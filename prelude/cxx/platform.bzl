# prelude/cxx/platform.bzl
#
# Extracted from buck2-prelude

load("@straylight_prelude//utils:platform_flavors_util.bzl", "by_platform")
load(":cxx_context.bzl", "get_cxx_platform_info")

def cxx_by_platform(ctx: AnalysisContext, xs: list[(str, typing.Any)]) -> list[typing.Any]:
    cxx_platform_info = get_cxx_platform_info(ctx)
    platform_flavors = [cxx_platform_info.name]
    if cxx_platform_info.deps_aliases:
        platform_flavors.extend(cxx_platform_info.deps_aliases)
    return by_platform(platform_flavors, xs)
