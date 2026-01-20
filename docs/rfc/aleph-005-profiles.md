# ℵ-005: Nix Invocation Profiles

| Status | Draft |
|--------|-------|
| Author | Straylight |
| Date | 2026-01-17 |

## Abstract

Define standardized invocation profiles (`nix-dev`, `nix-ci`, `nix-prod`) that set appropriate options for different contexts. The eval cache, while beneficial for CI and production, causes constant friction during rapid development.

## Motivation

### The Eval Cache Problem

Nix's evaluation cache (`~/.cache/nix/eval-cache-v5/`) caches the result of evaluating flake outputs. This is great for:

- CI pipelines (same commit = same eval)
- Production builds (reproducibility)

But it's **actively hostile** during development:

```bash
# Edit a file
vim nix/scripts/Aleph/Script/Config.hs

# Build - uses STALE cached evaluation
nix build .#foo
# ERROR: module not found (cache has old file list)

# Must remember the magic flag
nix build .#foo --no-eval-cache
# Works
```

The cache key includes the git tree hash, but:

1. Untracked files aren't in the tree hash
1. Unstaged changes aren't in the tree hash
1. Even staged changes can be stale if you're iterating fast

### Current Pain Points

1. **New files invisible**: Create `Foo.hs`, build fails because cache has old file list
1. **Edits invisible**: Modify `flake.nix`, evaluation uses cached result
1. **Inconsistent state**: Some derivations rebuild, others don't
1. **Cryptic errors**: "module not found" when the file clearly exists
1. **Mental overhead**: Must remember `--no-eval-cache` or waste time debugging

### The Broader Problem

Nix has many options that should differ by context:

| Option | Dev | CI | Prod |
|--------|-----|-----|------|
| `--no-eval-cache` | yes | no | no |
| `--show-trace` | yes | yes | no |
| `--print-build-logs` / `-L` | yes | yes | no |
| `--keep-failed` | yes | no | no |
| `--max-jobs` | auto | constrained | constrained |
| `--cores` | all | constrained | constrained |
| `--sandbox` | relaxed? | strict | strict |

Currently, developers must either:

- Remember to pass flags every time
- Set them globally in `nix.conf` (wrong for CI)
- Create shell aliases (not portable, not discoverable)

## Proposal

### 1. Invocation Wrapper Scripts

Create wrapper scripts that invoke `nix` with profile-appropriate options:

```bash
# nix-dev: Optimized for rapid iteration
nix-dev build .#foo
# Equivalent to: nix build .#foo --no-eval-cache --show-trace -L --keep-failed

# nix-ci: Optimized for CI pipelines  
nix-ci build .#foo
# Equivalent to: nix build .#foo --show-trace -L

# nix-prod: Production builds
nix-prod build .#foo
# Equivalent to: nix build .#foo
```

### 2. Implementation: Safe Bash + Haskell

Per ℵ-006 (Safe Bash), we split the implementation:

- **Bash**: Only environment setup + `exec` (zero logic)
- **Haskell**: All argument parsing and control flow

#### The Nix Wrapper (Safe Bash)

```nix
nix-dev = writeShellApplication {
  name = "nix-dev";
  runtimeInputs = [ ghcWithScript nix ];
  text = ''
    exec runghc -i${scriptSrc} ${scriptSrc}/nix-dev.hs "$@"
  '';
};
```

This is **safe bash**: no variables, no conditionals, just `exec`. The Nix interpolation (`${scriptSrc}`) produces static strings at build time.

#### The Logic (Haskell)

```haskell
-- nix/scripts/nix-dev.hs
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script
import System.Environment (getArgs)
import System.Posix.Process (executeFile)

main :: IO ()
main = do
  args <- getArgs
  
  let globalOpts = ["--no-eval-cache", "--show-trace"]
      buildOpts  = ["--print-build-logs", "--keep-failed"]
      
      extraOpts = case args of
        ("build":_)   -> globalOpts ++ buildOpts
        ("develop":_) -> globalOpts ++ buildOpts
        ("run":_)     -> globalOpts ++ buildOpts
        _             -> globalOpts
  
  executeFile "nix" True (args ++ extraOpts) Nothing
```

The `case` expression that would be a bug magnet in bash is now type-checked Haskell.

#### Why Not Pure Bash?

The bash version looks simpler:

```bash
case "${1:-}" in
  build|develop|run|shell)
    exec nix "$@" --no-eval-cache --show-trace -L --keep-failed
    ;;
  *)
    exec nix "$@" --no-eval-cache
    ;;
esac
```

But this has scope for bugs:

- `${1:-}` - easy to forget the `:-` and break on empty args
- `"$@"` position matters - putting it wrong breaks flag parsing
- Adding new cases requires careful quoting
- No type checking on the option strings

The Haskell version:

- Pattern matching is exhaustive and checked
- String lists are just lists - no quoting rules
- ~400ms interpreted startup (acceptable for a pre-build command)
- Maintainable by anyone who knows Haskell (our team)

#### Performance

```
$ time nix-dev --version
nix (Nix) 2.31.2+2

real    0m0.435s
```

~400ms is the `runghc` interpretation overhead. This is acceptable because:

1. It precedes a build that takes seconds to minutes
1. It's a developer tool, not a hot path
1. We could compile it if needed (drops to ~2ms)

### 3. Profile Definitions

#### `nix-dev` (Development)

Target: Local development, rapid iteration

```
--no-eval-cache      # Always re-evaluate (THE key option)
--show-trace         # Full error traces
--print-build-logs   # Stream build output (-L)
--keep-failed        # Keep failed build dirs for debugging
--log-format bar-with-logs  # Progress bar + logs
```

#### `nix-ci` (Continuous Integration)

Target: CI pipelines, PR checks

```
--show-trace         # Full error traces for debugging failures
--print-build-logs   # Capture all output
--log-format raw     # Machine-parseable output
# Note: eval cache ENABLED (same commit = same result)
```

#### `nix-prod` (Production)

Target: Release builds, deployment

```
# Minimal options - trust the defaults
--log-format raw
# eval cache ENABLED
# sandbox STRICT
```

### 4. Flake Module

Expose the wrappers via the flake:

```nix
# In aleph flake
packages.nix-dev = callPackage ./nix/scripts/nix-dev.nix {};
packages.nix-ci = callPackage ./nix/scripts/nix-ci.nix {};

# Or as a devShell addition
devShells.default = mkShell {
  packages = [ nix-dev nix-ci ];
  shellHook = ''
    alias nix='nix-dev'  # Optional: make dev the default
    echo "Using nix-dev profile (--no-eval-cache enabled)"
  '';
};
```

### 5. Documentation

Add to the flake template:

```markdown
## Nix Profiles

This project provides context-aware Nix wrappers:

- `nix-dev` - Development (no eval cache, verbose)
- `nix-ci` - CI pipelines (cached, verbose)
- `nix-prod` - Production (cached, quiet)

During development, use `nix-dev` to avoid eval cache staleness:

    nix-dev build .#myPackage

Or set it as your default in the dev shell.
```

## Alternatives Considered

### 1. Fix the Eval Cache

Ideally, Nix would:

- Invalidate cache when working tree changes (not just git tree hash)
- Detect untracked/modified files and warn or bypass cache
- Provide `--eval-cache=auto` that's smart about dirty trees

This requires upstream Nix changes. Our wrapper is a pragmatic workaround.

### 2. Global nix.conf Settings

```
# ~/.config/nix/nix.conf
eval-cache = false
```

Problems:

- Affects all projects (bad for large cached builds)
- Not portable across machines
- Can't easily toggle

### 3. Per-Project .nix.conf

Nix doesn't support per-project config files (unlike `.npmrc`, `.rustfmt.toml`, etc.).

### 4. Shell Aliases

```bash
alias nix='nix --no-eval-cache'
```

Problems:

- Not discoverable
- Not in version control
- Breaks scripts that call `nix` directly

## Implementation Plan

1. **Phase 1**: Create `nix-dev` wrapper in `nix/scripts/`
1. **Phase 2**: Add to devShell, document usage
1. **Phase 3**: Create `nix-ci` for GitHub Actions
1. **Phase 4**: Template integration

## Open Questions

1. **Should `nix-dev` be the default in devShells?**

   - Pro: Eliminates the most common friction
   - Con: Hides that it's non-standard behavior

1. **Should we warn when eval cache might be stale?**

   - Could check `git status --porcelain` and warn if dirty

1. **What about `nix develop`?**

   - The shell itself is cached - should we bust that too?

1. **Remote builders?**

   - `--no-eval-cache` is local; remote builders have their own cache

## References

- [ℵ-006: Safe Bash](aleph-006-safe-bash.md) - Defines the bash/Haskell boundary
- [ℵ-004: Aleph.Script](aleph-004-typed-unix.md) - The Haskell scripting infrastructure
- [Nix eval cache source](https://github.com/NixOS/nix/blob/master/src/libexpr/eval-cache.cc)
- [Related issue: eval cache with dirty trees](https://github.com/NixOS/nix/issues/6530)
- [Nix manual: common options](https://nixos.org/manual/nix/stable/command-ref/conf-file.html)
