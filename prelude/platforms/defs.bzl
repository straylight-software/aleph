# prelude/platforms/defs.bzl
#
# Platform detection and execution platform configuration.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Provides:
#   - execution_platform: rule to define an execution platform
#   - host_configuration: struct with cpu/os for current host
#
# Usage:
#   execution_platform(
#       name = "linux_x86_64",
#       cpu_configuration = "prelude//cpu:x86_64",
#       os_configuration = "prelude//os:linux",
#   )
#
#   # Or use host_configuration for default platform:
#   host_configuration.cpu  # "prelude//cpu:x86_64" on x86_64
#   host_configuration.os   # "prelude//os:linux" on Linux
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _execution_platform_impl(ctx: AnalysisContext) -> list[Provider]:
    constraints = dict()
    constraints.update(ctx.attrs.cpu_configuration[ConfigurationInfo].constraints)
    constraints.update(ctx.attrs.os_configuration[ConfigurationInfo].constraints)
    cfg = ConfigurationInfo(constraints = constraints, values = {})

    name = ctx.label.raw_target()
    platform = ExecutionPlatformInfo(
        label = name,
        configuration = cfg,
        executor_config = CommandExecutorConfig(
            local_enabled = ctx.attrs.local_enabled,
            remote_enabled = ctx.attrs.remote_enabled,
            use_windows_path_separators = ctx.attrs.use_windows_path_separators,
        ),
    )

    return [
        DefaultInfo(),
        platform,
        PlatformInfo(label = str(name), configuration = cfg),
        ExecutionPlatformRegistrationInfo(platforms = [platform]),
    ]

execution_platform = rule(
    impl = _execution_platform_impl,
    attrs = {
        "cpu_configuration": attrs.dep(providers = [ConfigurationInfo]),
        "os_configuration": attrs.dep(providers = [ConfigurationInfo]),
        "local_enabled": attrs.bool(default = True),
        "remote_enabled": attrs.bool(default = False),
        "use_windows_path_separators": attrs.bool(default = False),
    },
)

def _host_cpu_configuration() -> str:
    """Detect host CPU architecture."""
    arch = host_info().arch
    if arch.is_aarch64:
        return "prelude//cpu:arm64"
    elif arch.is_arm:
        return "prelude//cpu:arm32"
    elif arch.is_i386:
        return "prelude//cpu:x86_32"
    else:
        return "prelude//cpu:x86_64"

def _host_os_configuration() -> str:
    """Detect host operating system."""
    os = host_info().os
    if os.is_macos:
        return "prelude//os:macos"
    elif os.is_windows:
        return "prelude//os:windows"
    else:
        return "prelude//os:linux"

host_configuration = struct(
    cpu = _host_cpu_configuration(),
    os = _host_os_configuration(),
)
