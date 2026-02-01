# http_archive rule implementation
#
# Downloads and extracts an archive from a URL.

def _http_archive_impl(ctx: AnalysisContext) -> list[Provider]:
    """Download and extract an archive."""
    
    # Download the archive
    archive = ctx.actions.declare_output("_archive")
    url = ctx.attrs.urls[0]  # Just use first URL for now
    
    ctx.actions.download_file(
        archive,
        url,
        sha256 = ctx.attrs.sha256,
    )
    
    # Extract it
    output_dir = ctx.actions.declare_output("_extracted", dir = True)
    
    # Use tar to extract (works for .tar.gz, .tgz, .crate)
    strip_prefix = ctx.attrs.strip_prefix
    if strip_prefix:
        # Extract to temp, then move stripped content
        extract_cmd = cmd_args([
            "sh", "-c",
            cmd_args(
                "mkdir -p $2 && tar -xzf $1 -C $2 --strip-components=1",
                delimiter = " ",
            ),
            "--",
            archive,
            output_dir.as_output(),
        ])
    else:
        extract_cmd = cmd_args([
            "sh", "-c",
            cmd_args("mkdir -p $2 && tar -xzf $1 -C $2", delimiter = " "),
            "--",
            archive,
            output_dir.as_output(),
        ])
    
    ctx.actions.run(extract_cmd, category = "extract", identifier = ctx.label.name)
    
    return [DefaultInfo(default_output = output_dir)]

http_archive = rule(
    impl = _http_archive_impl,
    attrs = {
        "urls": attrs.list(attrs.string()),
        "sha256": attrs.option(attrs.string(), default = None),
        "strip_prefix": attrs.option(attrs.string(), default = None),
    },
)
