# nix/build/toolchains/haskell.bzl
#
# Haskell toolchain using GHC from Nix.
#
# Uses ghcWithPackages from the Nix devshell, which includes all
# dependencies. The bin/ghc wrapper filters Mercury-specific flags
# that stock GHC doesn't understand.
#
# Paths are read from .buckconfig.local [haskell] section.

load("@prelude//haskell:toolchain.bzl", "HaskellToolchainInfo", "HaskellPlatformInfo")

def _haskell_toolchain_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Haskell toolchain with paths from .buckconfig.local.

    Reads [haskell] section for:
      ghc              - GHC compiler
      ghc_pkg          - GHC package manager
      haddock          - Documentation generator
      ghc_lib_dir      - GHC library directory
      global_package_db - Global package database
    """

    ghc = read_root_config("haskell", "ghc", "bin/ghc")
    ghc_pkg = read_root_config("haskell", "ghc_pkg", "bin/ghc-pkg")
    haddock = read_root_config("haskell", "haddock", "bin/haddock")

    return [
        DefaultInfo(),
        HaskellToolchainInfo(
            compiler = ghc,
            packager = ghc_pkg,
            linker = ghc,
            haddock = haddock,
            compiler_flags = ctx.attrs.compiler_flags,
            linker_flags = ctx.attrs.linker_flags,
            ghci_script_template = ctx.attrs.ghci_script_template,
            ghci_iserv_template = ctx.attrs.ghci_iserv_template,
            script_template_processor = ctx.attrs.script_template_processor,
            packages = [],
            cache_links = True,
            archive_contents = "normal",
            support_expose_package = True,
        ),
        HaskellPlatformInfo(
            name = "x86_64-linux",
        ),
    ]

haskell_toolchain = rule(
    impl = _haskell_toolchain_impl,
    attrs = {
        "compiler_flags": attrs.list(attrs.string(), default = []),
        "linker_flags": attrs.list(attrs.string(), default = []),
        "ghci_script_template": attrs.option(attrs.source(), default = None),
        "ghci_iserv_template": attrs.option(attrs.source(), default = None),
        "script_template_processor": attrs.option(attrs.exec_dep(providers = [RunInfo]), default = None),
    },
    is_toolchain_rule = True,
)
