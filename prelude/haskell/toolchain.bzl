# prelude/haskell/toolchain.bzl
#
# Minimal Haskell toolchain providers.
#
# Extracted from buck2-prelude/haskell/toolchain.bzl (43 lines)
# Our version: just the provider definitions, no baggage.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The upstream prelude haskell/ directory contains:
#
#   toolchain.bzl      (43 lines)   - Provider definitions [EXTRACTED HERE]
#   library_info.bzl   (45 lines)   - HaskellLibraryInfo provider
#   link_info.bzl      (46 lines)   - HaskellLinkInfo provider
#   util.bzl           (151 lines)  - Shared utilities
#   compile.bzl        (266 lines)  - Compilation logic
#   haskell.bzl        (1161 lines) - Main rules (haskell_binary, haskell_library)
#   haskell_ghci.bzl   (732 lines)  - GHCi REPL support [INTERESTING]
#   haskell_haddock.bzl (160 lines) - Documentation generation [INTERESTING]
#   haskell_ide.bzl    (9 lines)    - IDE integration stub [INTERESTING]
#   ide/               (dir)        - IDE tooling
#   tools/             (dir)        - Build tool scripts
#
# What's worth keeping (to rewrite in Haskell later):
#   - GHCi support: loading projects in REPL with correct deps
#   - Haddock: documentation generation
#   - IDE: hie-bios / HLS integration
#   - Template Haskell: splice handling, staging
#
# What's noise:
#   - Complex link ordering logic (GHC handles this)
#   - Profiling variants (rebuild with -prof is fine)
#   - archive_contents modes (we use GHC's defaults)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HaskellPlatformInfo = provider(fields = {
    "name": provider_field(typing.Any, default = None),
})

HaskellToolchainInfo = provider(
    fields = {
        # Core tools
        "compiler": provider_field(typing.Any, default = None),        # ghc
        "packager": provider_field(typing.Any, default = None),        # ghc-pkg
        "linker": provider_field(typing.Any, default = None),          # ghc (usually same as compiler)
        "haddock": provider_field(typing.Any, default = None),         # haddock

        # Flags
        "compiler_flags": provider_field(typing.Any, default = None),
        "linker_flags": provider_field(typing.Any, default = None),

        # GHCi support [INTERESTING - rewrite target]
        "ghci_script_template": provider_field(typing.Any, default = None),
        "ghci_iserv_template": provider_field(typing.Any, default = None),
        "ghci_binutils_path": provider_field(typing.Any, default = None),
        "ghci_lib_path": provider_field(typing.Any, default = None),
        "ghci_ghc_path": provider_field(typing.Any, default = None),
        "ghci_iserv_path": provider_field(typing.Any, default = None),
        "ghci_iserv_prof_path": provider_field(typing.Any, default = None),
        "ghci_cxx_path": provider_field(typing.Any, default = None),
        "ghci_cc_path": provider_field(typing.Any, default = None),
        "ghci_cpp_path": provider_field(typing.Any, default = None),
        "ghci_packager": provider_field(typing.Any, default = None),

        # IDE support [INTERESTING - rewrite target]
        "ide_script_template": provider_field(typing.Any, default = None),

        # Misc
        "compiler_major_version": provider_field(typing.Any, default = None),
        "package_name_prefix": provider_field(typing.Any, default = None),
        "support_always_use_cache": provider_field(bool, default = False),
        "use_argsfile": provider_field(typing.Any, default = None),
        "support_expose_package": provider_field(bool, default = False),
        "archive_contents": provider_field(typing.Any, default = None),
        "cache_links": provider_field(typing.Any, default = None),
        "script_template_processor": provider_field(typing.Any, default = None),
    },
)
