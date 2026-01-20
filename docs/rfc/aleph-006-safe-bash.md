# ℵ-006: Safe Bash

| Status | Draft |
|--------|-------|
| Author | Straylight |
| Date | 2026-01-17 |

## Abstract

Define "safe bash" as bash code with zero scope for bugs: no variables, no interpolation, no control flow, no logic. Just `exec`. This establishes a clear boundary between acceptable and unacceptable shell usage in Straylight.

## The Problem

ℵ-004 (Aleph.Script) argues for replacing bash with typed Haskell. But we still use bash in places like:

```bash
# nix/overlays/script.nix
text = ''
  exec runghc -i${scriptSrc} ${scriptSrc}/nix-dev.hs "$@"
'';
```

Is this hypocrisy? No. This is **safe bash**.

## Definition

**Safe bash** is bash code where:

1. **No variables** - Nothing is stored, compared, or manipulated
1. **No interpolation** - No `$var`, `${var}`, `$(cmd)` in runtime bash
1. **No control flow** - No `if`, `case`, `for`, `while`, `&&`, `||`
1. **No logic** - No decisions, no branches, no computation
1. **Only exec** - The entire script is `exec something "$@"`

The only "interpolation" permitted is Nix string interpolation at *build time*, which produces static strings baked into the derivation.

## Examples

### Safe Bash

```bash
# Just exec with passthrough
exec runghc -i/nix/store/xxx-scripts /nix/store/xxx-scripts/foo.hs "$@"
```

```bash
# Setting environment then exec (no runtime variables)
export PATH="/nix/store/xxx/bin:$PATH"
exec /nix/store/yyy/bin/program "$@"
```

```bash
# cd then exec
cd /some/fixed/path
exec ./run "$@"
```

The key insight: **`"$@"` is the only permitted variable**, and it's a passthrough - we never inspect, modify, or branch on it.

### Unsafe Bash (Prohibited)

```bash
# Variable assignment - UNSAFE
IMAGE="${1:-ubuntu:24.04}"

# Conditional - UNSAFE  
if [ -z "$GPU_ADDR" ]; then
  GPU_ADDR=$(detect_gpu)
fi

# Loop - UNSAFE
for cmd in sh mount hostname; do
  ln -sf busybox "$cmd"
done

# String manipulation - UNSAFE
DRIVER_DIR="${NVPATH%/bin/nvidia-smi}"

# Command substitution in logic - UNSAFE
if [ "$(id -u)" -ne 0 ]; then
  exec sudo "$0" "$@"
fi
```

## Why This Boundary?

### The Bug Surface Area Argument

Bash bugs come from:

- Unquoted variables: `$var` vs `"$var"`
- Word splitting: `$var` on values with spaces
- Glob expansion: `$var` on values with `*`
- Empty variables: `${var:-default}` forgotten
- Exit code handling: `set -e` subtleties
- Subshell scoping: `x=1; (x=2); echo $x`
- Array syntax: `"${arr[@]}"` vs `${arr[*]}`
- Interpolation in interpolation: `"${var/foo/$bar}"`

**Safe bash has zero surface area for any of these bugs.**

When the only variable is `"$@"` passed directly to `exec`:

- No quoting bugs (it's always quoted)
- No word splitting (exec handles it)
- No glob expansion (no glob context)
- No empty variable bugs (passthrough doesn't care)
- No exit code handling (exec replaces the process)
- No scoping (no subshells)
- No arrays (just `$@`)
- No nested interpolation (no interpolation at all)

### The Nix Interpolation Distinction

```nix
text = ''
  exec ${pkgs.ripgrep}/bin/rg "$@"
'';
```

This looks like interpolation, but it's not *bash* interpolation. The Nix evaluator produces:

```bash
exec /nix/store/abc123-ripgrep/bin/rg "$@"
```

The "variable" `${pkgs.ripgrep}` doesn't exist at bash runtime. It's a static string. This is the same as writing the path literally - just more maintainable.

**Rule: Nix interpolation into bash is safe iff it produces only static strings.**

Unsafe Nix interpolation:

```nix
# UNSAFE - generates bash variable assignment
text = ''
  CPUS="${toString cfg.cpus}"
  if [ "$CPUS" -gt 4 ]; then ...
'';
```

Safe Nix interpolation:

```nix
# SAFE - static strings only, exec passthrough
text = ''
  exec ${program}/bin/prog --cpus ${toString cfg.cpus} "$@"
'';
```

## The Wrapper Pattern

Safe bash exists precisely for one use case: **environment setup before exec**.

Nix's `writeShellApplication` and `makeWrapper` produce scripts that:

1. Set up `PATH`
1. Set environment variables
1. `exec` the real program

This is irreducible. You can't `exec` from Haskell into a modified environment without *something* setting up that environment. That something is either:

- A bash wrapper (what we do)
- A C wrapper (what makeWrapper does internally)
- Kernel environment inheritance (requires parent setup)

We choose bash because:

- Nix already generates it
- It's human-readable for debugging
- It's trivially auditable when it's just `exec`

## Enforcement

### In Code Review

Any bash in the codebase should be auditable in < 5 seconds:

- Is it just `exec ... "$@"`? ✓ Safe
- Does it have `if`, `case`, `for`, `while`? ✗ Rewrite in Haskell
- Does it assign variables? ✗ Rewrite in Haskell
- Does it do command substitution `$()`? ✗ Rewrite in Haskell

### In Nix Modules

```nix
# GOOD: Safe bash wrapper
myTool = writeShellApplication {
  name = "my-tool";
  runtimeInputs = [ ghcWithScript ];
  text = ''
    exec runghc -i${scriptSrc} ${scriptSrc}/my-tool.hs "$@"
  '';
};

# BAD: Logic in bash
myTool = writeShellApplication {
  name = "my-tool";
  text = ''
    case "$1" in        # ← Logic in bash, should be in Haskell
      build) ... ;;
      run) ... ;;
    esac
  '';
};
```

### Automated Checking

A safe-bash linter could verify:

```
safe-bash-check() {
  # Must match: optional env setup, then exec
  grep -qE '^(export [A-Z_]+="[^"$]*"\n)*(cd [^\n]+\n)?exec ' "$1"
}
```

This is intentionally restrictive. If your bash doesn't match, rewrite it.

## Relationship to Other RFCs

- **ℵ-004 (Aleph.Script)**: Defines where logic goes (Haskell)
- **ℵ-005 (Nix Profiles)**: Example of safe bash + Haskell pattern
- **ℵ-006 (this)**: Defines the boundary between them

The architecture:

```
┌─────────────────────────────────────────────────────┐
│                    Nix Module                        │
│  writeShellApplication {                            │
│    text = "exec runghc ... script.hs \"$@\"";       │
│  }                 │                                │
│                    │ safe bash (just exec)          │
│                    ▼                                │
│  ┌─────────────────────────────────────────────┐   │
│  │              Haskell Script                  │   │
│  │  - Argument parsing                          │   │
│  │  - Control flow                              │   │
│  │  - Error handling                            │   │
│  │  - All logic                                 │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## FAQ

### What about `set -euo pipefail`?

Safe bash doesn't need it. There's nothing to fail except `exec`, and if `exec` fails, the shell exits with the error anyway.

But `writeShellApplication` adds it automatically, and it doesn't hurt.

### What about `trap`?

If you need cleanup, you need logic. Rewrite in Haskell with `bracket`.

### What about checking if a file exists?

That's logic. Rewrite in Haskell with `test_f`.

### What about simple conditionals like `[ -n "$DEBUG" ] && set -x`?

Still logic, still scope for bugs (`-n` vs `-z` mixup, quoting). If you want debug output, pass a flag to the Haskell script.

### Isn't this too restrictive?

Yes, intentionally. The goal is a bright line that's easy to audit. "Is this just exec?" is a question anyone can answer in one second.

## Conclusion

Safe bash is bash with zero scope for bugs. It's the irreducible shell that sets up the environment for typed programs. Everything else belongs in Haskell.

The mantra: **If it has a branch, it's not bash.**
