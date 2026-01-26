# prelude/toolchains/genrule.bzl
#
# System genrule toolchain.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Provides a no-op genrule toolchain. The upstream version can optionally
# provide a zip_scrubber for deterministic zip outputs.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load("@prelude//:genrule_toolchain.bzl", "GenruleToolchainInfo")

def _system_genrule_toolchain_impl(_ctx):
    return [
        DefaultInfo(),
        GenruleToolchainInfo(
            zip_scrubber = None,
        ),
    ]

system_genrule_toolchain = rule(
    impl = _system_genrule_toolchain_impl,
    attrs = {},
    is_toolchain_rule = True,
)
