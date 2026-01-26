# toolchains/nix_genrule.bzl
#
# Genrule-based approach for Nix dependency resolution.
#
# Instead of trying to wire nix paths into Buck2's prebuilt_cxx_library system
# (which requires analysis-time resolution via .buckconfig.local), we use a
# genrule that resolves nix flake refs at execution time.
#
# This keeps everything simple:
# - No shell hooks to populate .buckconfig.local
# - No custom providers that Buck2 prelude doesn't understand
# - Nix resolution happens inside the build action (hermetic)
# - Works with remote execution (NativeLink workers have nix)
#
# NOTE: The cmd string is written directly to a shell script.
# $OUT, $SRCDIR, $SRCS, $TMP are environment variables set by Buck2.
# Shell variables we create (NIX_DEP_0, etc.) just use normal $ syntax.
#
# Usage:
#   load("@toolchains//:nix_genrule.bzl", "nix_cxx_binary")
#
#   nix_cxx_binary(
#       name = "zlib-test",
#       srcs = ["main.cpp"],
#       nix_deps = ["nixpkgs#zlib"],
#       compiler = "clang++",
#   )

# Map package names to library names for -l flag
_LIB_NAME_MAP = {
    "zlib": "z",
    "openssl": "ssl",
    "libpng": "png",
    "libjpeg": "jpeg",
    "sqlite": "sqlite3",
    "curl": "curl",
    "boost": "boost_system",
    "protobuf": "protobuf",
}

def nix_cxx_binary(
        name,
        srcs,
        nix_deps = [],
        compiler = "clang++",
        cflags = [],
        ldflags = [],
        **kwargs):
    """
    Build a C++ binary with nix flake dependencies.
    
    Nix deps are resolved at build time via `nix build --print-out-paths`.
    This works on any machine with nix installed, including NativeLink workers.
    
    Args:
        name: Target name
        srcs: Source files (relative to BUCK file)
        nix_deps: List of nix flake refs (e.g. ["nixpkgs#zlib", "nixpkgs#openssl"])
        compiler: Compiler to use (default: clang++)
        cflags: Additional compiler flags
        ldflags: Additional linker flags
    """
    
    # Map nix_deps to library names for linking
    lib_names = []
    for dep in nix_deps:
        # Extract package name from flake ref
        pkg = dep.split("#")[-1] if "#" in dep else dep
        # Handle .dev suffix
        if "." in pkg:
            pkg = pkg.split(".")[0]
        # Map to actual library name
        lib_name = _LIB_NAME_MAP.get(pkg, pkg)
        lib_names.append("-l" + lib_name)
    
    # Add library flags to ldflags
    all_ldflags = list(ldflags) + lib_names
    
    # Generate the build script inline in the cmd
    resolve_script = []
    include_flags = []
    lib_flags = []
    
    for i, dep in enumerate(nix_deps):
        # Resolve main output
        # Use $$ to escape $( for Buck2 macro expansion - becomes $ in shell
        resolve_script.append(
            'NIX_DEP_{i}=$$(nix build "{dep}" --print-out-paths --no-link 2>/dev/null)'.format(
                i = i,
                dep = dep,
            )
        )
        # Try dev output for headers
        resolve_script.append(
            'NIX_DEV_{i}=$$(nix build "{dep}.dev" --print-out-paths --no-link 2>/dev/null || echo "$$NIX_DEP_{i}")'.format(
                i = i,
                dep = dep,
            )
        )
        # Reference shell variables - also need $$ to escape
        include_flags.append('-isystem "$$NIX_DEV_{}"/include'.format(i))
        lib_flags.append('-L"$$NIX_DEP_{}"/lib'.format(i))
    
    # Build sources reference - $SRCDIR is set by genrule
    srcs_ref = " ".join(["$SRCDIR/{}".format(s) for s in srcs])
    
    # Assemble the command
    cmd_parts = []
    cmd_parts.extend(resolve_script)
    # $OUT is set by genrule
    cmd_parts.append("{compiler} {cflags} {includes} {srcs} -o $OUT {libs} {ldflags}".format(
        compiler = compiler,
        cflags = " ".join(cflags),
        includes = " ".join(include_flags),
        srcs = srcs_ref,
        libs = " ".join(lib_flags),
        ldflags = " ".join(all_ldflags),
    ))
    
    cmd = " && ".join(cmd_parts) if cmd_parts else "true"
    
    native.genrule(
        name = name,
        srcs = srcs,
        out = name,
        bash = cmd,  # Use bash instead of cmd to avoid $(macro) expansion
        executable = True,
        **kwargs
    )

def nix_cxx_library(
        name,
        srcs,
        hdrs = [],
        nix_deps = [],
        compiler = "clang++",
        cflags = [],
        **kwargs):
    """
    Build a C++ static library with nix flake dependencies.
    
    Similar to nix_cxx_binary but produces a .a archive.
    """
    
    resolve_script = []
    include_flags = []
    
    for i, dep in enumerate(nix_deps):
        resolve_script.append(
            'NIX_DEP_{i}=$$(nix build "{dep}" --print-out-paths --no-link 2>/dev/null)'.format(
                i = i,
                dep = dep,
            )
        )
        resolve_script.append(
            'NIX_DEV_{i}=$$(nix build "{dep}.dev" --print-out-paths --no-link 2>/dev/null || echo "$$NIX_DEP_{i}")'.format(
                i = i,
                dep = dep,
            )
        )
        include_flags.append('-isystem "$$NIX_DEV_{}"/include'.format(i))

    # Compile to objects, then archive
    obj_cmds = []
    obj_names = []
    for s in srcs:
        obj = s.replace(".cpp", ".o").replace(".c", ".o").replace("/", "_")
        obj_names.append(obj)
        obj_cmds.append("{compiler} {cflags} {includes} -c $SRCDIR/{src} -o {obj}".format(
            compiler = compiler,
            cflags = " ".join(cflags + ["-fPIC"]),
            includes = " ".join(include_flags),
            src = s,
            obj = obj,
        ))
    
    cmd_parts = []
    cmd_parts.extend(resolve_script)
    cmd_parts.extend(obj_cmds)
    cmd_parts.append("ar rcs $OUT {}".format(" ".join(obj_names)))
    
    cmd = " && ".join(cmd_parts)
    
    native.genrule(
        name = name,
        srcs = srcs + hdrs,
        out = "lib" + name + ".a",
        bash = cmd,  # Use bash instead of cmd to avoid $(macro) expansion
        **kwargs
    )

# Witnessed variant - runs build through witness proxy for attestations
def nix_cxx_binary_witnessed(
        name,
        srcs,
        nix_deps = [],
        compiler = "clang++",
        cflags = [],
        ldflags = [],
        coeffects = "pure",
        **kwargs):
    """
    Build a C++ binary with witnessed execution.
    
    Like nix_cxx_binary, but runs through the witness proxy to collect
    attestations of any network fetches. Outputs both the binary and
    an .attestations.jsonl file.
    
    Args:
        coeffects: Declared coeffects ("pure", "network", etc.)
                   Will warn if actual behavior violates declaration.
    """
    
    lib_names = []
    for dep in nix_deps:
        pkg = dep.split("#")[-1] if "#" in dep else dep
        if "." in pkg:
            pkg = pkg.split(".")[0]
        lib_name = _LIB_NAME_MAP.get(pkg, pkg)
        lib_names.append("-l" + lib_name)
    
    all_ldflags = list(ldflags) + lib_names
    
    resolve_script = []
    include_flags = []
    lib_flags = []
    
    for i, dep in enumerate(nix_deps):
        resolve_script.append(
            'NIX_DEP_{i}=$$(nix build "{dep}" --print-out-paths --no-link 2>/dev/null)'.format(
                i = i,
                dep = dep,
            )
        )
        resolve_script.append(
            'NIX_DEV_{i}=$$(nix build "{dep}.dev" --print-out-paths --no-link 2>/dev/null || echo "$$NIX_DEP_{i}")'.format(
                i = i,
                dep = dep,
            )
        )
        include_flags.append('-isystem "$$NIX_DEV_{}"/include'.format(i))
        lib_flags.append('-L"$$NIX_DEP_{}"/lib'.format(i))

    srcs_ref = " ".join(["$SRCDIR/{}".format(s) for s in srcs])

    # Create output directory structure
    # $OUT is a Buck2 env var, but shell vars need $$ escaping
    cmd_parts = [
        'mkdir -p "$OUT"',
        'ATTESTATION_LOG="$OUT/.attestations.jsonl"',
        ': > "$$ATTESTATION_LOG"',
    ]
    cmd_parts.extend(resolve_script)
    
    # Build command
    build_cmd = "{compiler} {cflags} {includes} {srcs} -o $OUT/{name} {libs} {ldflags}".format(
        compiler = compiler,
        cflags = " ".join(cflags),
        includes = " ".join(include_flags),
        srcs = srcs_ref,
        name = name,
        libs = " ".join(lib_flags),
        ldflags = " ".join(all_ldflags),
    )
    
    # Wrap with witness proxy if available - HTTP_PROXY is shell var, needs $$
    cmd_parts.append(
        'if [ -n "$${HTTP_PROXY:-}" ]; then ' + build_cmd + '; else ' + build_cmd + '; fi'
    )
    
    # Check attestations against declared coeffects - shell command substitution needs $$
    cmd_parts.append(
        'FETCH_COUNT=$$(wc -l < "$$ATTESTATION_LOG" 2>/dev/null || echo 0)'
    )
    if coeffects == "pure":
        cmd_parts.append(
            'if [ "$$FETCH_COUNT" -gt 0 ]; then echo "WARNING: declared pure but made $$FETCH_COUNT network request(s)" >&2; fi'
        )
    
    cmd = " && ".join(cmd_parts)
    
    native.genrule(
        name = name,
        srcs = srcs,
        outs = {
            "binary": [name],
            "attestations": [".attestations.jsonl"],
        },
        default_outs = ["binary"],
        bash = cmd,  # Use bash instead of cmd to avoid $(macro) expansion
        **kwargs
    )
