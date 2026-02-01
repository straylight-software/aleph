# prelude/os_lookup/defs.bzl
#
# OS detection and platform lookup.
# Used by rules to determine target OS and generate appropriate scripts.
#
# Extracted from buck2-prelude/os_lookup/defs.bzl
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The os_lookup module provides:
#   - Os enum: all supported operating systems
#   - ScriptLanguage enum: shell script types (sh, bat)
#   - OsLookup provider: runtime OS detection
#   - os_lookup rule: creates OsLookup provider
#
# Used by:
#   - genrule: to determine script language for commands
#   - sh_binary: to select appropriate shell
#   - Platform configuration
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Operating systems - includes all platforms Buck2 supports
Os = enum(
    # Standard platforms
    "linux",
    "macos",
    "windows",
    "freebsd",
    # Special values
    "fat_mac_linux",  # Universal Mac/Linux builds
    "unknown",
)

# Script language for genrule/sh_binary
ScriptLanguage = enum(
    "sh",   # Unix shell script (bash/sh)
    "bat",  # Windows batch file
)

# Provider for OS information at analysis time
OsLookup = provider(fields = {
    # CPU architecture (x86_64, arm64, etc.) or None
    "cpu": provider_field(str | None, default = None),
    # Operating system
    "os": provider_field(Os, default = Os("unknown")),
    # Script language to use for this OS
    "script": provider_field(ScriptLanguage, default = ScriptLanguage("sh")),
})

def _os_lookup_impl(ctx: AnalysisContext) -> list[Provider]:
    """Implementation of os_lookup rule."""
    return [
        DefaultInfo(),
        OsLookup(
            cpu = ctx.attrs.cpu,
            os = Os(ctx.attrs.os),
            script = ScriptLanguage(ctx.attrs.script),
        ),
    ]

# Rule to create an OsLookup provider for a specific platform configuration
os_lookup = rule(
    impl = _os_lookup_impl,
    attrs = {
        "cpu": attrs.option(attrs.string(), default = None),
        "os": attrs.enum(Os.values()),
        "script": attrs.enum(ScriptLanguage.values()),
    },
)

# Convenience functions for common OS checks
def is_linux(os: Os) -> bool:
    return os == Os("linux")

def is_macos(os: Os) -> bool:
    return os == Os("macos")

def is_windows(os: Os) -> bool:
    return os == Os("windows")

def is_unix(os: Os) -> bool:
    """Returns True for Unix-like systems (Linux, macOS, FreeBSD)."""
    return os in (Os("linux"), Os("macos"), Os("freebsd"), Os("fat_mac_linux"))
