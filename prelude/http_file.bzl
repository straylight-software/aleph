# http_file rule implementation
#
# Extracted from buck2-prelude http_file.bzl
# Stripped: materialization_test dependency, vpnless_urls

load("@straylight_prelude//utils:expect.bzl", "expect")
load("@straylight_prelude//utils:utils.bzl", "value_or")

def http_file_shared(
        actions: AnalysisActions,
        name: str,
        url: str,
        is_executable: bool,
        sha1: [None, str],
        sha256: [None, str],
        size_bytes: [None, int]) -> list[Provider]:
    """Shared implementation for http_file downloads."""
    output = actions.declare_output(name)
    actions.download_file(
        output,
        url,
        is_executable = is_executable,
        sha1 = sha1,
        sha256 = sha256,
        size_bytes = size_bytes,
    )

    providers = [DefaultInfo(default_output = output)]
    if is_executable:
        providers.append(RunInfo(args = [output]))
    return providers

def http_file_impl(ctx: AnalysisContext) -> list[Provider]:
    """Implementation of the http_file build rule."""
    expect(len(ctx.attrs.urls) == 1, "multiple `urls` not supported: {}", ctx.attrs.urls)
    return http_file_shared(
        ctx.actions,
        name = value_or(ctx.attrs.out, ctx.label.name),
        url = ctx.attrs.urls[0],
        sha1 = ctx.attrs.sha1,
        sha256 = ctx.attrs.sha256,
        is_executable = ctx.attrs.executable or False,
        size_bytes = ctx.attrs.size_bytes,
    )
