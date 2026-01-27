# prelude/utils/platform_flavors_util.bzl
#
# Extracted from buck2-prelude

def by_platform(
        platform_flavors: list[str],
        xs: list[(str, typing.Any)]) -> list[typing.Any]:
    """
    Resolve platform-flavor-specific parameters, given the list of platform
    flavors to match against.
    """
    res = []
    for (dtype, deps) in xs:
        for platform in platform_flavors:
            if regex_match(dtype, platform):
                res.append(deps)
    return res
