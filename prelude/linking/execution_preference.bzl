# prelude/linking/execution_preference.bzl
#
# Link execution preference for local vs remote execution.
# Controls where link actions run (local, remote, hybrid).
#
# Extracted from buck2-prelude/linking/execution_preference.bzl (82 lines)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Link execution preferences control where link actions run:
#   - any: No preference (use executor default)
#   - full_hybrid: Run both locally and remotely
#   - local: Prefer local if compatible
#   - local_only: Must run locally, error otherwise
#   - remote: Prefer remote if available
#
# This is useful for:
#   - Large binaries that don't cache well on RE
#   - Build stamping that requires local execution
#   - Platform-specific linking requirements
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LinkExecutionPreferenceTypes = [
    "any",
    "full_hybrid",
    "local",
    "local_only",
    "remote",
]

LinkExecutionPreference = enum(*LinkExecutionPreferenceTypes)

# Provider for custom execution preference determination
LinkExecutionPreferenceDeterminatorInfo = provider(fields = {
    # Function: (links: list[Label], deps_prefs: list[LinkExecutionPreferenceInfo]) -> LinkExecutionPreference
    "preference_for_links": provider_field(typing.Any, default = None),
})

# Provider carrying a target's link execution preference
LinkExecutionPreferenceInfo = provider(fields = {
    "preference": provider_field(typing.Any, default = None),
})

# Record for action execution attributes derived from preference
ActionExecutionAttributes = record(
    full_hybrid = field(bool, default = False),
    local_only = field(bool, default = False),
    prefer_local = field(bool, default = False),
    prefer_remote = field(bool, default = False),
)

def link_execution_preference_attr():
    """
    Attribute for link execution preference on rules.
    
    Options:
    - any: No preference, use buck2's executor configuration
    - full_hybrid: Execute both locally and remotely
    - local: Execute locally if compatible
    - local_only: Execute locally, error if incompatible
    - remote: Execute remotely if possible, else locally
    """
    return attrs.option(
        attrs.one_of(
            attrs.enum(LinkExecutionPreferenceTypes),
            attrs.dep(providers = [LinkExecutionPreferenceDeterminatorInfo]),
        ),
        default = None,
    )

def get_link_execution_preference(ctx: AnalysisContext, links: list[Label]) -> LinkExecutionPreference:
    """Get the resolved link execution preference for a target."""
    if not hasattr(ctx.attrs, "link_execution_preference"):
        fail("`get_link_execution_preference` called on a rule that does not support link_execution_preference!")

    link_execution_preference = ctx.attrs.link_execution_preference

    # If no preference has been set, default to any
    if not link_execution_preference:
        return LinkExecutionPreference("any")

    # Direct enum value
    if not isinstance(link_execution_preference, Dependency):
        return LinkExecutionPreference(link_execution_preference)

    # Determinator dependency - not implemented in minimal extraction
    # Would need cxx_attr_deps/cxx_attr_exported_deps
    fail("LinkExecutionPreferenceDeterminatorInfo not supported in minimal prelude")

def get_action_execution_attributes(preference: LinkExecutionPreference) -> ActionExecutionAttributes:
    """Convert a LinkExecutionPreference to action execution attributes."""
    if preference == LinkExecutionPreference("any"):
        return ActionExecutionAttributes()
    elif preference == LinkExecutionPreference("full_hybrid"):
        return ActionExecutionAttributes(full_hybrid = True)
    elif preference == LinkExecutionPreference("local"):
        return ActionExecutionAttributes(prefer_local = True)
    elif preference == LinkExecutionPreference("local_only"):
        return ActionExecutionAttributes(local_only = True)
    elif preference == LinkExecutionPreference("remote"):
        return ActionExecutionAttributes(prefer_remote = True)
    else:
        fail("Unhandled LinkExecutionPreference: {}".format(str(preference)))
