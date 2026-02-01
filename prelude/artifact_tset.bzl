# prelude/artifact_tset.bzl
#
# Extracted from buck2-prelude

load("@straylight_prelude//utils:expect.bzl", "expect")
load("@straylight_prelude//utils:utils.bzl", "flatten")

ArtifactInfoTag = enum(
    "swiftmodule",
    "swift_pcm",
)

ArtifactInfo = record(
    label = field(Label),
    artifacts = field(list[Artifact]),
    tags = field(list[ArtifactInfoTag]),
)

def stringify_artifact_label(value: Label | str) -> str:
    if type(value) == "string":
        return value
    return str(value.raw_target())

def _get_artifacts(entries: list[ArtifactInfo]) -> list[Artifact]:
    return flatten([entry.artifacts for entry in entries])

_ArtifactTSet = transitive_set(
    args_projections = {
        "artifacts": _get_artifacts,
    },
)

ArtifactTSet = record(
    _tset = field([_ArtifactTSet, None], None),
)

def make_artifact_tset(
        actions: AnalysisActions,
        label: Label | None = None,
        artifacts: list[Artifact] = [],
        infos: list[ArtifactInfo] = [],
        children: list[ArtifactTSet] = [],
        tags: list[ArtifactInfoTag] = []) -> ArtifactTSet:
    expect(
        label != None or not artifacts,
        "must pass in `label` to associate with artifacts",
    )

    children = [c._tset for c in children if c._tset != None]

    values = []
    if artifacts:
        values.append(ArtifactInfo(label = label, artifacts = artifacts, tags = tags))
    values.extend(infos)

    if not values and not children:
        return ArtifactTSet()

    kwargs = {}
    if values:
        kwargs["value"] = values
    if children:
        kwargs["children"] = children
    return ArtifactTSet(
        _tset = actions.tset(_ArtifactTSet, **kwargs),
    )

def project_artifacts(
        actions: AnalysisActions,
        tsets: list[ArtifactTSet] = []) -> list[TransitiveSetArgsProjection]:
    tset = make_artifact_tset(
        actions = actions,
        children = tsets,
    )

    if tset._tset == None:
        return []

    return [tset._tset.project_as_args("artifacts")]
