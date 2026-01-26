# prelude/rust/rust_toolchain.bzl
#
# Minimal extraction from buck2-prelude/rust/rust_toolchain.bzl (145 lines)
# We need PanicRuntime enum and RustToolchainInfo provider.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream rust_toolchain.bzl contains:
#   - RustExplicitSysrootDeps record (explicit sysroot handling)
#   - PanicRuntime enum (unwind, abort, none)
#   - rust_toolchain_attrs dict (all the toolchain fields)
#   - RustToolchainInfo provider
#
# What's worth keeping (to rewrite in Haskell later):
#   - Sysroot management for cross-compilation
#   - Clippy integration
#   - Rustdoc integration
#
# What's noise:
#   - Most of the attrs are optional with sensible defaults
#   - The upstream version has extensive comments we can trim
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Explicit sysroot deps for nostd/custom targets
RustExplicitSysrootDeps = record(
    core = Dependency | None,
    proc_macro = Dependency | None,
    std = Dependency | None,
    panic_unwind = Dependency | None,
    panic_abort = Dependency | None,
    others = list[Dependency],
)

PanicRuntime = enum("unwind", "abort", "none")

# Toolchain attributes - trimmed to essentials
rust_toolchain_attrs = {
    # Core tools
    "compiler": provider_field(RunInfo | None, default = None),
    "rustdoc": provider_field(RunInfo | None, default = None),
    "clippy_driver": provider_field(RunInfo | None, default = None),

    # Target configuration
    "rustc_target_triple": provider_field(str | None, default = None),
    "default_edition": provider_field(str | None, default = None),
    "panic_runtime": provider_field(PanicRuntime),

    # Flags
    "rustc_flags": provider_field(list[typing.Any], default = []),
    "extra_rustc_flags": provider_field(list[typing.Any], default = []),
    "rustc_check_flags": provider_field(list[typing.Any], default = []),
    "rustc_binary_flags": provider_field(list[typing.Any], default = []),
    "rustc_test_flags": provider_field(list[typing.Any], default = []),
    "rustc_coverage_flags": provider_field(typing.Any, default = ("-Cinstrument-coverage",)),
    "linker_flags": provider_field(list[typing.Any], default = []),

    # Rustdoc
    "rustdoc_env": provider_field(dict[str, typing.Any], default = {}),
    "rustdoc_flags": provider_field(list[typing.Any], default = []),
    "doctests": provider_field(bool, default = False),

    # Lints
    "allow_lints": provider_field(list[typing.Any], default = []),
    "deny_lints": provider_field(list[typing.Any], default = []),
    "warn_lints": provider_field(list[typing.Any], default = []),
    "deny_on_check_lints": provider_field(list[typing.Any], default = []),
    "clippy_toml": provider_field(Artifact | None, default = None),

    # Sysroot
    "sysroot_path": provider_field(Artifact | None, default = None),
    "explicit_sysroot_deps": provider_field(RustExplicitSysrootDeps | None, default = None),

    # Advanced
    "advanced_unstable_linking": provider_field(bool, default = False),
    "nightly_features": provider_field(bool, default = False),
    "report_unused_deps": provider_field(bool, default = False),
    "rust_target_path": provider_field(Dependency | None, default = None),
    "configuration_hash": provider_field(str | None, default = None),

    # Tools
    "llvm_lines_tool": provider_field(RunInfo | None, default = None),
    "measureme_crox": provider_field(RunInfo | None, default = None),
    "make_trace_upload": provider_field(typing.Callable[[Artifact], RunInfo] | None, default = None),
    "rust_error_handler": provider_field(typing.Any, default = None),
}

RustToolchainInfo = provider(fields = rust_toolchain_attrs)
