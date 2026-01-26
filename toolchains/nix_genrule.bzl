# toolchains/nix_genrule.bzl
#
# Nix-native build rules for Buck2.
#
# The contract:
#   - Any derivation is a valid dep (nixpkgs#foo, /flake#bar, etc.)
#   - nix-analyze resolves the derivation and extracts compiler flags
#   - Buck2 runs the build action with resolved paths
#   - DICE tracks the dep graph for incremental rebuilds
#
# Usage:
#   load("@toolchains//:nix_genrule.bzl", "nix_binary")
#
#   nix_binary(
#       name = "demo",
#       srcs = ["main.cpp"],
#       deps = ["nixpkgs#simdjson", "nixpkgs#zlib"],
#   )

NixResolveInfo = provider(fields = ["flags"])

def _nix_resolve_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Resolve nix flake refs to compiler flags.
    
    Calls nix-analyze at execution time to get -isystem, -L, -l flags.
    """
    out = ctx.actions.declare_output("flags.txt")
    
    # Build deps as space-separated list (no extra quotes needed - nix handles #)
    deps_list = " ".join(ctx.attrs.deps)
    
    # Build shell script as a single cmd_args with spaces
    # Format: /path/to/nix-analyze resolve dep1 dep2 > /path/to/out
    script = cmd_args(
        ctx.attrs._analyzer[RunInfo],
        "resolve",
        deps_list,
        ">",
        out.as_output(),
        delimiter = " ",
    )
    
    ctx.actions.run(
        cmd_args("sh", "-c", script),
        category = "nix_resolve",
        identifier = ctx.attrs.name,
        local_only = True,
    )
    
    return [
        DefaultInfo(default_output = out),
        NixResolveInfo(flags = out),
    ]

nix_resolve = rule(
    impl = _nix_resolve_impl,
    attrs = {
        "deps": attrs.list(attrs.string(), default = [], doc = "Nix flake refs"),
        "_analyzer": attrs.exec_dep(default = "root//src/armitage:nix-analyze"),
    },
)

def _nix_binary_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Build a binary with Nix derivation dependencies.
    """
    out = ctx.actions.declare_output(ctx.attrs.name)
    
    # Get compiler from config
    compiler = read_root_config("cxx", "cxx", "clang++")
    
    # Build toolchain include flags from config
    toolchain_flags = []
    
    clang_resource_dir = read_root_config("cxx", "clang_resource_dir", None)
    if clang_resource_dir:
        toolchain_flags.append("-resource-dir=" + clang_resource_dir)
        toolchain_flags.append("-isystem" + clang_resource_dir + "/include")
    
    gcc_include = read_root_config("cxx", "gcc_include", None)
    if gcc_include:
        toolchain_flags.append("-isystem" + gcc_include)
    
    gcc_include_arch = read_root_config("cxx", "gcc_include_arch", None)
    if gcc_include_arch:
        toolchain_flags.append("-isystem" + gcc_include_arch)
    
    glibc_include = read_root_config("cxx", "glibc_include", None)
    if glibc_include:
        toolchain_flags.append("-isystem" + glibc_include)
    
    # Build link flags from config
    link_flags = ["-fuse-ld=lld"]
    
    glibc_lib = read_root_config("cxx", "glibc_lib", None)
    if glibc_lib:
        link_flags.extend(["-B" + glibc_lib, "-L" + glibc_lib, "-Wl,-rpath," + glibc_lib])
    
    gcc_lib = read_root_config("cxx", "gcc_lib", None)
    if gcc_lib:
        link_flags.extend(["-B" + gcc_lib, "-L" + gcc_lib, "-Wl,-rpath," + gcc_lib])
    
    gcc_lib_base = read_root_config("cxx", "gcc_lib_base", None)
    if gcc_lib_base:
        link_flags.extend(["-L" + gcc_lib_base, "-Wl,-rpath," + gcc_lib_base])
    
    # Build compiler flags
    cflags = toolchain_flags[:]
    if ctx.attrs.std != "":
        cflags.append("-std=" + ctx.attrs.std)
    cflags.extend(ctx.attrs.extra_cflags)
    
    # Build the compile command
    cmd = cmd_args()
    cmd.add(compiler)
    cmd.add(cflags)
    
    # Combine all link flags
    all_ldflags = link_flags + ctx.attrs.extra_ldflags
    
    # Add nix-resolved flags if we have deps
    if ctx.attrs.nix_deps:
        flags_file = ctx.attrs.nix_deps[NixResolveInfo].flags
        # Read flags from file and pass to compiler
        script = cmd_args(
            "sh", "-c",
            cmd_args(
                compiler, " ",
                " ".join(cflags), " ",
                "$(cat ", flags_file, ") ",
                cmd_args(ctx.attrs.srcs, delimiter = " "), " ",
                "-o ", out.as_output(), " ",
                " ".join(all_ldflags),
                delimiter = "",
            ),
        )
        ctx.actions.run(
            script,
            category = "nix_compile",
            identifier = ctx.attrs.name,
            local_only = True,
        )
    else:
        cmd = cmd_args(compiler)
        cmd.add(cflags)
        cmd.add(ctx.attrs.srcs)
        cmd.add("-o", out.as_output())
        cmd.add(all_ldflags)
        ctx.actions.run(cmd, category = "compile", identifier = ctx.attrs.name, local_only = True)
    
    return [
        DefaultInfo(default_output = out),
        RunInfo(args = cmd_args(out)),
    ]

nix_binary = rule(
    impl = _nix_binary_impl,
    attrs = {
        "srcs": attrs.list(attrs.source(), default = []),
        "nix_deps": attrs.option(attrs.dep(providers = [NixResolveInfo]), default = None),
        "std": attrs.string(default = "c++20"),
        "extra_cflags": attrs.list(attrs.string(), default = []),
        "extra_ldflags": attrs.list(attrs.string(), default = []),
    },
)

# ════════════════════════════════════════════════════════════════════════════════
# Macro wrapper for convenience
# ════════════════════════════════════════════════════════════════════════════════

def nix_cxx_binary(
        name,
        srcs,
        deps = [],
        std = "c++20",
        extra_cflags = [],
        extra_ldflags = [],
        visibility = None):
    """
    Convenience macro that creates both the resolver and binary.
    
    Usage:
        nix_cxx_binary(
            name = "demo",
            srcs = ["main.cpp"],
            deps = ["nixpkgs#simdjson"],
        )
    """
    if deps:
        nix_resolve(
            name = name + "_deps",
            deps = deps,
        )
        nix_binary(
            name = name,
            srcs = srcs,
            nix_deps = ":" + name + "_deps",
            std = std,
            extra_cflags = extra_cflags,
            extra_ldflags = extra_ldflags,
            visibility = visibility,
        )
    else:
        nix_binary(
            name = name,
            srcs = srcs,
            std = std,
            extra_cflags = extra_cflags,
            extra_ldflags = extra_ldflags,
            visibility = visibility,
        )
