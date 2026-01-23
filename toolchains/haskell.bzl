# nix/build/toolchains/haskell.bzl
#
# Haskell toolchain using GHC from Nix.
#
# Uses ghcWithPackages from the Nix devshell, which includes all
# dependencies. The bin/ghc wrapper filters Mercury-specific flags
# that stock GHC doesn't understand.
#
# Paths are read from .buckconfig.local [haskell] section.
#
# Rules:
#   haskell_toolchain - toolchain definition
#   haskell_ffi_binary - Haskell binary with C/C++ FFI

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
            cache_links = True,
            archive_contents = "normal",
            support_expose_package = False,
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

# =============================================================================
# haskell_ffi_binary - Haskell binary with C/C++ FFI
# =============================================================================

def _haskell_ffi_binary_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Build a Haskell binary that calls C/C++ code via FFI.
    
    Steps:
      1. Compile C++ sources to .o files with clang
      2. Compile and link Haskell sources with GHC, including the C++ objects
    """
    # Get tools from config
    ghc = read_root_config("haskell", "ghc", "ghc")
    cxx = read_root_config("cxx", "cxx", "clang++")
    
    # C++ stdlib paths for unwrapped clang
    gcc_include = read_root_config("cxx", "gcc_include", "")
    gcc_include_arch = read_root_config("cxx", "gcc_include_arch", "")
    glibc_include = read_root_config("cxx", "glibc_include", "")
    clang_resource_dir = read_root_config("cxx", "clang_resource_dir", "")
    
    # Library paths for linking
    gcc_lib = read_root_config("cxx", "gcc_lib", "")
    gcc_lib_base = read_root_config("cxx", "gcc_lib_base", "")
    glibc_lib = read_root_config("cxx", "glibc_lib", "")
    
    # Output binary
    out = ctx.actions.declare_output(ctx.attrs.name)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Compile C++ sources to object files
    # ─────────────────────────────────────────────────────────────────────────
    cxx_compile_flags = [
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-c",
    ]
    
    # Add stdlib paths for unwrapped clang
    if gcc_include:
        cxx_compile_flags.extend(["-isystem", gcc_include])
    if gcc_include_arch:
        cxx_compile_flags.extend(["-isystem", gcc_include_arch])
    if glibc_include:
        cxx_compile_flags.extend(["-isystem", glibc_include])
    if clang_resource_dir:
        cxx_compile_flags.extend(["-resource-dir=" + clang_resource_dir])
    
    # Add include path for headers (current source directory)
    cxx_compile_flags.extend(["-I", "."])
    
    cxx_objects = []
    for src in ctx.attrs.cxx_srcs:
        obj_name = src.short_path.replace(".cpp", ".o").replace(".c", ".o")
        obj = ctx.actions.declare_output(obj_name)
        
        cmd = cmd_args([cxx] + cxx_compile_flags + [
            "-o", obj.as_output(),
            src,
        ])
        
        ctx.actions.run(cmd, category = "cxx_compile", identifier = src.short_path)
        cxx_objects.append(obj)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Compile Haskell and link with C++ objects
    # ─────────────────────────────────────────────────────────────────────────
    ghc_flags = [
        "-O2",
        "-threaded",  # Enable threaded runtime for FFI
    ]
    
    # Add library path for C++ stdlib (GHC's Nix wrapper handles most paths)
    if gcc_lib_base:
        ghc_flags.extend(["-optl", "-L" + gcc_lib_base])
    
    # Link C++ stdlib
    ghc_flags.extend(["-lstdc++"])
    
    # Build GHC command
    ghc_cmd = cmd_args([ghc] + ghc_flags + [
        "-o", out.as_output(),
    ])
    
    # Add Haskell sources
    for src in ctx.attrs.hs_srcs:
        ghc_cmd.add(src)
    
    # Add C++ object files
    for obj in cxx_objects:
        ghc_cmd.add(obj)
    
    ctx.actions.run(ghc_cmd, category = "ghc_link")
    
    return [
        DefaultInfo(default_output = out),
        RunInfo(args = [out]),
    ]

haskell_ffi_binary = rule(
    impl = _haskell_ffi_binary_impl,
    attrs = {
        "hs_srcs": attrs.list(attrs.source()),
        "cxx_srcs": attrs.list(attrs.source(), default = []),
        "cxx_headers": attrs.list(attrs.source(), default = []),
        "deps": attrs.list(attrs.dep(), default = []),
        "compiler_flags": attrs.list(attrs.string(), default = []),
    },
)

# =============================================================================
# haskell_script - Simple Haskell binary for single-file scripts
# =============================================================================
#
# Unlike haskell_binary (from prelude), this rule:
# - Compiles and links in one GHC invocation
# - Properly handles deps on haskell_library (uses package-db)
# - Works with single-file Main modules (common for scripts)
# - Uses ghcWithPackages from Nix for external deps
#
# For multi-module binaries with complex deps, use haskell_binary from prelude.

def _haskell_script_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Build a single-file Haskell script with library dependencies.
    
    Compiles and links in one GHC invocation. Instead of linking against
    pre-built library .a files (which would require managing complex link
    order for transitive deps), we use GHC's -i flag to include library
    source directories, allowing GHC to compile everything together.
    
    This is simpler and more reliable for scripts that use a single library.
    For complex multi-library setups, use the prelude's haskell_binary.
    """
    ghc = read_root_config("haskell", "ghc", "bin/ghc")
    
    out = ctx.actions.declare_output(ctx.attrs.name)
    
    ghc_cmd = cmd_args([ghc])
    
    # Add compiler flags
    ghc_cmd.add(ctx.attrs.compiler_flags)
    
    # Add output
    ghc_cmd.add("-o", out.as_output())
    
    # Add source include paths for library deps
    # We use -i<dir> which tells GHC to search there for imported modules
    for include_path in ctx.attrs.include_paths:
        ghc_cmd.add("-i" + include_path)
    
    # Add external package dependencies (from ghcWithPackages)
    external_packages = ctx.attrs.packages
    for pkg in external_packages:
        ghc_cmd.add("-package", pkg)
    
    # Add source files
    for src in ctx.attrs.srcs:
        ghc_cmd.add(src)
    
    ctx.actions.run(ghc_cmd, category = "haskell_script")
    
    return [
        DefaultInfo(default_output = out),
        RunInfo(args = [out]),
    ]

haskell_script = rule(
    impl = _haskell_script_impl,
    attrs = {
        "srcs": attrs.list(attrs.source()),
        "include_paths": attrs.list(attrs.string(), default = []),
        "compiler_flags": attrs.list(attrs.string(), default = []),
        "packages": attrs.list(attrs.string(), default = []),
    },
)
