# toolchains/nix_analyze.bzl
#
# Buck2 rules that use the nix-analyze Haskell tool.
#
# The Haskell analyzer handles all nix resolution logic:
#   - Resolving flake refs to store paths
#   - Mapping package names to library names
#   - Handling multiple outputs (.out, .dev)
#
# This keeps the Starlark simple and the nix logic testable.
#
# Usage:
#   load("@toolchains//:nix_analyze.bzl", "nix_library", "nix_cxx_binary")
#
#   nix_library(
#       name = "zlib",
#       flake_ref = "nixpkgs#zlib",
#   )
#
#   nix_cxx_binary(
#       name = "my-app",
#       srcs = ["main.cpp"],
#       deps = [":zlib"],
#   )

# Provider for nix-resolved library info
NixLibraryInfo = provider(fields = [
    "flake_ref",      # Original flake reference
])

def _nix_library_impl(ctx):
    """
    Expose a nix package as a C++ library.
    
    This is a lightweight rule - actual resolution happens in nix_cxx_binary
    via the Haskell analyzer.
    """
    flake_ref = ctx.attrs.flake_ref
    
    # Marker file
    marker = ctx.actions.write(
        "nix_lib.txt",
        ["flake_ref: " + flake_ref],
    )
    
    return [
        DefaultInfo(default_output = marker),
        NixLibraryInfo(flake_ref = flake_ref),
    ]

nix_library = rule(
    impl = _nix_library_impl,
    attrs = {
        "flake_ref": attrs.string(),
    },
)

def _nix_cxx_binary_impl(ctx):
    """
    Build a C++ binary using nix-resolved dependencies.
    
    Uses the Haskell nix-analyze tool to resolve all nix deps,
    then compiles with the resolved flags.
    """
    srcs = ctx.attrs.srcs
    deps = ctx.attrs.deps
    name = ctx.attrs.name
    
    # Read compiler paths from .buckconfig.local
    cxx = read_root_config("cxx", "cxx", "clang++")
    clang_resource_dir = read_root_config("cxx", "clang_resource_dir", "")
    gcc_include = read_root_config("cxx", "gcc_include", "")
    gcc_include_arch = read_root_config("cxx", "gcc_include_arch", "")
    glibc_include = read_root_config("cxx", "glibc_include", "")
    gcc_lib = read_root_config("cxx", "gcc_lib", "")
    gcc_lib_base = read_root_config("cxx", "gcc_lib_base", "")
    glibc_lib = read_root_config("cxx", "glibc_lib", "")
    
    # Collect nix flake refs from deps
    flake_refs = []
    for dep in deps:
        if NixLibraryInfo in dep:
            flake_refs.append(dep[NixLibraryInfo].flake_ref)
    
    out = ctx.actions.declare_output(name)
    
    # System include flags
    system_includes = []
    if clang_resource_dir:
        system_includes.extend(["-resource-dir", clang_resource_dir])
    if gcc_include:
        system_includes.extend(["-isystem", gcc_include])
    if gcc_include_arch:
        system_includes.extend(["-isystem", gcc_include_arch])
    if glibc_include:
        system_includes.extend(["-isystem", glibc_include])
    
    # System library paths
    system_libs = []
    if gcc_lib:
        system_libs.extend(["-B", gcc_lib, "-L", gcc_lib])
    if gcc_lib_base:
        system_libs.extend(["-L", gcc_lib_base])
    if glibc_lib:
        system_libs.extend(["-B", glibc_lib, "-L", glibc_lib])
    
    # Get the analyzer tool - we pass it via env var so the script can find it
    analyzer_run_info = ctx.attrs._analyzer[RunInfo]
    
    # Write a wrapper script that:
    # 1. Calls nix-analyze to get flags
    # 2. Compiles with those flags
    #
    # We pass the analyzer path via $ANALYZER env var since we can't
    # interpolate cmd_args artifacts directly into script strings.
    script_lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Resolve nix dependencies using Haskell analyzer",
    ]
    
    if flake_refs:
        # Call analyzer to get flags - $ANALYZER is set via env
        refs_str = " ".join(flake_refs)
        script_lines.append('NIX_FLAGS=$("$ANALYZER" resolve {})'.format(refs_str))
    else:
        script_lines.append("NIX_FLAGS=")
    
    script_lines.append("")
    script_lines.append("# Compile")
    
    src_refs = " ".join(['"$SRCDIR/{}"'.format(s.short_path) for s in srcs])
    
    compile_cmd = '{cxx} {sys_includes} {cflags} $NIX_FLAGS {srcs} -o "$OUT" {sys_libs}'.format(
        cxx = cxx,
        sys_includes = " ".join(system_includes),
        cflags = " ".join(ctx.attrs.cflags),
        srcs = src_refs,
        sys_libs = " ".join(system_libs),
    )
    script_lines.append(compile_cmd)
    
    script = ctx.actions.write(
        "build.sh",
        script_lines,
        is_executable = True,
    )
    
    # Symlinked source directory
    srcs_dir = ctx.actions.symlinked_dir(
        "srcs",
        {src.short_path: src for src in srcs},
    )
    
    # Build cmd_args for the analyzer path
    # The RunInfo.args contains the executable, we extract it for the env var
    analyzer_cmd = cmd_args(analyzer_run_info.args)
    
    ctx.actions.run(
        cmd_args(["/bin/bash", "-e", script], hidden = [srcs_dir, analyzer_cmd]),
        env = {
            "OUT": out.as_output(),
            "SRCDIR": srcs_dir,
            "ANALYZER": analyzer_cmd,
        },
        category = "nix_cxx_compile",
        local_only = True,
    )
    
    return [
        DefaultInfo(default_output = out),
        RunInfo(args = cmd_args(out)),
    ]

nix_cxx_binary = rule(
    impl = _nix_cxx_binary_impl,
    attrs = {
        "srcs": attrs.list(attrs.source()),
        "deps": attrs.list(attrs.dep()),
        "cflags": attrs.list(attrs.string(), default = []),
        "_analyzer": attrs.exec_dep(default = "root//src/armitage:nix-analyze"),
    },
)
