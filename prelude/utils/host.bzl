# Host OS detection utilities
#
# Extracted from buck2-prelude utils/host.bzl

HostOSTypes = ["linux", "macos", "windows"]

HostOSType = enum(*HostOSTypes)

def _compute_get_host_os() -> HostOSType:
    info = host_info()
    if info.os.is_linux:
        return HostOSType("linux")
    elif info.os.is_macos:
        return HostOSType("macos")
    elif info.os.is_windows:
        return HostOSType("windows")
    else:
        fail("Unknown host OS")

_HOST_OS = _compute_get_host_os()

def get_host_os() -> HostOSType:
    """Get the host operating system type."""
    return _HOST_OS
