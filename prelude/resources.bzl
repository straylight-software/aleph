# prelude/resources.bzl
#
# Resource management for C++ and Rust binaries.
# Provides ResourceInfo provider for transitive resource gathering.
#
# Extracted from buck2-prelude/resources.bzl (53 lines)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Resources are data files bundled with binaries:
#   - Config files, assets, data files
#   - Shared between C++ and Rust rules
#   - Accessed at runtime via resource DB
#
# The implementation:
#   - ResourceInfo provider: tracks transitive resources
#   - gather_resources: collects resources from deps
#   - create_resource_db: generates resource lookup DB for binary
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load("@straylight_prelude//:artifacts.bzl", "ArtifactOutputs")

# Resources provider for transitive deps, shared by C++ and Rust.
ResourceInfo = provider(fields = {
    # A map containing all resources from transitive dependencies.
    # Keys are rule labels, values are maps of resource names to artifacts.
    "resources": provider_field(dict[Label, dict[str, ArtifactOutputs]]),
})

def gather_resources(
        label: Label,
        resources: dict[str, ArtifactOutputs] = {},
        deps: list[Dependency] = []) -> dict[Label, dict[str, ArtifactOutputs]]:
    """
    Return the resources for this rule and its transitive deps.
    """
    all_resources = {}

    # Resources for self
    if resources:
        all_resources[label] = resources

    # Merge in resources for deps
    for dep in deps:
        if ResourceInfo in dep:
            all_resources.update(dep[ResourceInfo].resources)

    return all_resources

def create_resource_db(
        ctx: AnalysisContext,
        name: str,
        binary: Artifact,
        resources: dict[str, ArtifactOutputs]) -> Artifact:
    """
    Generate a resource DB for resources for the given binary,
    relativized to the binary's working directory.
    """
    db = {
        name: cmd_args(resource.default_output, delimiter = "", relative_to = (binary, 1))
        for (name, resource) in resources.items()
    }
    return ctx.actions.write_json(name, db)
