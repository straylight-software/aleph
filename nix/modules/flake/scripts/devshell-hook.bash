echo "━━━ aleph-naught devshell ━━━"

# GHC packages are baked into ghcWithPackages - no runtime config needed
# Buck2 uses its own GHC from .buckconfig.local with explicit -package flags
echo "GHC: $("@ghcWithAllDeps@"/bin/ghc --version)"
@ghcWasmCheck@
@straylightNixCheck@
# Buck2 build system integration
@buildShellHook@

# Shortlist hermetic C++ libraries
@shortlistShellHook@

# Local Remote Execution (NativeLink)
@lreShellHook@

@extraShellHook@
