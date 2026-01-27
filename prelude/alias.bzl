# Alias rule implementations
#
# Extracted from buck2-prelude alias.bzl

def alias_impl(ctx: AnalysisContext) -> list[Provider]:
    """Implementation of the alias build rule."""
    if ctx.attrs.actual:
        return ctx.attrs.actual.providers
    else:
        return [DefaultInfo()]

def configured_alias_impl(ctx: AnalysisContext) -> list[Provider]:
    """Implementation of the configured_alias build rule."""
    if ctx.attrs.configured_actual != None and ctx.attrs.fallback_actual != None:
        fail("cannot set both of `configured_actual` and `fallback_actual`")
    if ctx.attrs.configured_actual != None:
        return ctx.attrs.configured_actual.providers
    if ctx.attrs.fallback_actual != None:
        return ctx.attrs.fallback_actual.providers
    fail("must set one of `configured_actual` or `fallback_actual`")

def versioned_alias_impl(_ctx: AnalysisContext) -> list[Provider]:
    # Should be intercepted in macro stub and converted to `alias`.
    fail("unsupported")
