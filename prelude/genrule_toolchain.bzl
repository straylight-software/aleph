# prelude/genrule_toolchain.bzl
#
# Minimal extraction from buck2-prelude/genrule_toolchain.bzl
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream genrule_toolchain.bzl defines the GenruleToolchainInfo provider.
# It's used to provide tools for genrule execution (zip scrubber, etc).
#
# We only need the provider definition - the system_genrule_toolchain rule
# in toolchains/ provides a no-op implementation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GenruleToolchainInfo = provider(fields = {
    "zip_scrubber": provider_field(typing.Any, default = None),
})
