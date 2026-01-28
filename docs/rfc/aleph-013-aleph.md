# RFC-013: Aleph - Unified Build Compiler

| Field | Value |
|-------|-------|
| RFC | aleph-013 |
| Title | Aleph - Unified Build Compiler |
| Author | b7r6 |
| Status | Draft |
| Created | 2026-01-28 |
| Supersedes | RFC-011, RFC-012 (render.nix) |
| Extends | RFC-008 (Continuity) |

## Abstract

Aleph is a compiler. The frontend (`render`) parses Nix, infers types, validates
layout, and emits metadata. The backend (`armitage`) consumes that metadata to
build, cache, and attest without the nix-daemon.

The daemon is the problem. It's a trusted intermediary that controls store
access, makes substitution decisions, runs builders, and signs paths. DICE can
replace this if we have a complete static graph extractable before Nix
evaluation.

This RFC defines the contract between frontend and backend: **if render succeeds
with no violations, armitage can build without the daemon**.

## Motivation

### The Daemon Problem

`nix-daemon` is in the critical path for everything:

```
nix build → daemon → eval → instantiate → build → sign → store
                ↑
            trusted root
```

Every build requires the daemon. Every substitution. Every signature. The daemon
is a single point of trust, a single point of failure, and a massive surface
area.

### The Solution: Static Extraction

If we can extract the complete build graph **before** evaluation:

```
render parse → lint → infer → layout → FlakeMetadata
                                            ↓
                                      armitage analyze
                                            ↓
                                      ActionGraph (DICE)
                                            ↓
                                      execute → attest
                                            ↓
                                      /nix/store (direct write)
```

No daemon in the loop. The build graph is content-addressed. Each step is
attested. The store is just bytes.

### Why This Requires Language Restrictions

Certain Nix constructs make static extraction impossible:

| Construct | Why It Breaks Static Analysis |
|-----------|------------------------------|
| `with expr;` | Scope not statically resolvable |
| `rec { }` | Enables cycles, breaks topo-sort |
| IFD | Requires eval to discover graph |
| `import <nixpkgs>` | Non-reproducible path |
| `"string".attr` | hnix parser rejects it |

These aren't style preferences. They're **the contract** between render and
armitage. Ban them, and you get daemon-free attested builds.

## Architecture

### The Compiler Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                              aleph                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FRONTEND (render)                  BACKEND (armitage)              │
│  ─────────────────                  ──────────────────              │
│                                                                     │
│  ┌─────────┐                                                        │
│  │  Parse  │  Nix source → AST (hnix)                               │
│  └────┬────┘                                                        │
│       ↓                                                             │
│  ┌─────────┐                                                        │
│  │  Lint   │  Detect with/rec/IFD/banned syntax                     │
│  └────┬────┘                                                        │
│       ↓                                                             │
│  ┌─────────┐                                                        │
│  │  Infer  │  Hindley-Milner type inference                         │
│  └────┬────┘                                                        │
│       ↓                                                             │
│  ┌─────────┐                                                        │
│  │ Layout  │  Directory structure → module graph                    │
│  └────┬────┘                                                        │
│       ↓                                                             │
│  ┌─────────┐         ┌─────────────────────────────────────────┐   │
│  │Metadata │────────→│            FlakeMetadata                 │   │
│  └─────────┘         │  - inputs (pinned)                       │   │
│                      │  - modules (typed, linted)               │   │
│                      │  - packages                              │   │
│                      │  - graph (import edges)                  │   │
│                      │  - violations (must be empty)            │   │
│                      └──────────────┬──────────────────────────┘   │
│                                     ↓                               │
│                              ┌─────────┐                            │
│                              │ Analyze │  Metadata → ActionGraph    │
│                              └────┬────┘                            │
│                                   ↓                                 │
│                              ┌─────────┐                            │
│                              │  DICE   │  Incremental computation   │
│                              └────┬────┘                            │
│                                   ↓                                 │
│                              ┌─────────┐                            │
│                              │ Execute │  Local or Remote (RE)      │
│                              └────┬────┘                            │
│                                   ↓                                 │
│                              ┌─────────┐                            │
│                              │ Attest  │  Sign outputs, record      │
│                              └────┬────┘                            │
│                                   ↓                                 │
│                              ┌─────────┐                            │
│                              │  Store  │  Direct write, no daemon   │
│                              └─────────┘                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### FlakeMetadata: The Contract

```haskell
-- The output of render, input to armitage
data FlakeMetadata = FlakeMetadata
  { fmInputs     :: Map Text FlakeRef       -- Pinned flake inputs
  , fmModules    :: Map (Kind, Name) Module -- All modules, typed
  , fmPackages   :: Map Name FilePath       -- Package definitions
  , fmOverlays   :: Map Name FilePath       -- Overlay definitions
  , fmChecks     :: Map Name FilePath       -- Check definitions
  , fmLib        :: Map Name FilePath       -- Library functions
  , fmGraph      :: ModuleGraph             -- Import dependency graph
  , fmTypes      :: Map FilePath NixType    -- Inferred types
  , fmViolations :: [Violation]             -- MUST be empty
  }

-- A violation blocks the pipeline
data Violation
  = VWith FilePath Span         -- with expr;
  | VRec FilePath Span          -- rec { }
  | VParseFail FilePath Text    -- hnix rejected it
  | VBareCommand FilePath Span Text  -- unresolved command (bash)
  | VHeredoc FilePath Span      -- heredoc in bash
  | VIfd FilePath Span          -- import from derivation
  | VMissingClass FilePath      -- module without _class
  | VBadLayout FilePath Text    -- layout violation
```

### The Contract

**If `fmViolations == []`, armitage can build without the daemon.**

This is not advisory. This is the compilation contract. Violations are errors,
not warnings. There is no `--force`. Fix the code.

## Flake Layout

### Directory-as-Type

```
flake.nix                 # Inputs + mkFlake, explicit imports
nix/
  modules/
    flake/
      <name>.nix          # FlakeModule, _class = "flake"
    nixos/
      <name>.nix          # NixOSModule, _class = "nixos"
    home/
      <name>.nix          # HomeManagerModule, _class = "home"
  overlays/
    <name>.nix            # final -> prev -> { }
  packages/
    <name>.nix            # { pkgs, ... }: derivation
  lib/
    <name>.nix            # Pure functions
  checks/
    <name>.nix            # { pkgs, ... }: derivation
```

### Rules

| Rule | Rationale |
|------|-----------|
| No `_index.nix` | Graph derived from directory scan |
| No `_main.nix` | Explicit imports in flake.nix |
| One module per file | Filename = export name |
| `_class` required | Kind validation |
| Subdirs allowed | `build/` → `build.nix` or `build/default.nix` |

### Composite Modules

Modules may import other modules:

```nix
# nix/modules/flake/full.nix
{
  _class = "flake";
  
  imports = [
    ./build.nix
    ./shortlist.nix
    ./lre.nix
    ./devshell.nix
    ./nixpkgs.nix
  ];
}
```

This is fine. The import graph is statically extractable.

### Generated Index

The index is computed, not written:

```nix
# Derived by render, not hand-maintained
flake.modules.flake = {
  build = import ./nix/modules/flake/build.nix;
  devshell = import ./nix/modules/flake/devshell.nix;
  full = import ./nix/modules/flake/full.nix;
  # ... discovered from directory
};
```

## Banned Constructs

### Nix

| Construct | Code | Why |
|-----------|------|-----|
| `with expr;` | N001 | Scope not statically resolvable |
| `rec { }` | N002 | Enables cycles, breaks topo-sort |
| `"string".attr` | - | hnix parser rejects it |
| IFD | N003 | Requires eval to discover graph |
| `import <nixpkgs>` | N004 | Non-reproducible |

### Bash (in Nix)

| Construct | Code | Why |
|-----------|------|-----|
| Heredocs `<<` | E001 | Escaping hell, untyped |
| Here-strings `<<<` | E002 | Same issues |
| `eval` | E003 | Unanalyzable |
| Backticks | E004 | Deprecated |
| Bare commands | - | Non-reproducible |

### No Escape Hatch

There is no `# aleph-ignore`. There is no `--force`. There is no override.

The restrictions exist because without them, **static extraction is impossible**.
If you need `with` or `rec`, you cannot use aleph. Use the daemon.

## Type System

### Nix Types

```haskell
data NixType
  = TVar TypeVar           -- Inference variable
  | TInt | TFloat | TBool | TString | TPath | TNull
  | TList NixType
  | TAttrs (Map Text NixType)
  | TAttrsOpen (Map Text NixType)  -- May have more fields
  | TFun NixType NixType
  | TDerivation
  | TUnion [NixType]
  | TAny                   -- Top type
```

### Inference Sources

| Source | Example | Inferred |
|--------|---------|----------|
| Literals | `42` | `Int` |
| Defaults | `{ port ? 8080 }` | `port : Int` |
| Operators | `a + b` where `a : Int` | `b : Int` |
| Builtins | `map f xs` | `f : a -> b`, `xs : [a]` |
| Application | `f x` where `f : a -> b` | `x : a` |

### Typed Builtins

40+ builtins typed:

```
map        : (a -> b) -> [a] -> [b]
filter     : (a -> Bool) -> [a] -> [a]
toString   : a -> String
attrNames  : {} -> [String]
hasAttr    : String -> {} -> Bool
derivation : { ... } -> Derivation
```

### Bash Type Inference

```haskell
data BashType = TInt | TString | TBool | TPath | TArray BashType
```

Inference from patterns:

| Pattern | Inferred |
|---------|----------|
| `${VAR:-8080}` | `VAR : Int` |
| `${VAR:-true}` | `VAR : Bool` |
| `${VAR:-hello}` | `VAR : String` |
| `${VAR:?}` | `VAR : String`, required |
| `${VAR:-$OTHER}` | `VAR : typeof(OTHER)` |
| `curl --timeout $VAR` | `VAR : Int` (from builtins) |

## CLI

### Unified Interface

```bash
# Validation (frontend only)
aleph check .              # Parse, lint, infer, layout
aleph check --graph .      # Show module graph
aleph check --types .      # Show inferred types
aleph check --violations . # Show violations (non-zero exit)

# Building (frontend + backend)
aleph build .              # Full pipeline
aleph build --remote .     # Via Remote Execution
aleph build --attest .     # With attestation

# Legacy compatibility
aleph run nixpkgs#hello    # Translates to nix (for now)
aleph shell nixpkgs#rust   # Translates to nix (for now)
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success, no violations |
| 1 | Violations found |
| 2 | Parse failure |
| 3 | Build failure |

### Output Formats

```bash
aleph check --format=human .  # Default, human-readable
aleph check --format=json .   # Machine-readable
aleph check --format=sarif .  # GitHub Code Scanning
```

## Integration with Continuity

### The Pipeline

```
render (this RFC)
    │
    │ FlakeMetadata
    ▼
armitage (RFC-008)
    │
    │ ActionGraph
    ▼
DICE (RFC-008)
    │
    │ Execute + Cache
    ▼
Store (RFC-008)
    │
    │ Content-addressed
    ▼
Attestation (RFC-008)
```

### What render Provides

1. **Static graph** - All imports known before eval
2. **Typed modules** - Every binding has a type
3. **No violations** - Graph is safe for DICE

### What armitage Consumes

1. **FlakeMetadata** - The complete, validated graph
2. **Store paths** - Resolved from flake refs
3. **Types** - For generating typed configs

### The Escape Plan

```
Phase 1 (current):
  Nix → render → armitage → daemon
                    ↑
              validation only

Phase 2 (this RFC):
  Nix → render → armitage → NativeLink
                    ↑
              daemon bypass

Phase 3 (future):
  Dhall → armitage → DICE → R2
           ↑
        no Nix evaluator
```

## Implementation

### Existing Code

```
nix/render/
├── app/render.hs           # CLI
├── lib/Render/
│   ├── Nix/
│   │   ├── Parse.hs        # hnix wrapper
│   │   ├── Infer.hs        # HM inference
│   │   ├── Types.hs        # NixType
│   │   ├── Lint.hs         # with/rec detection
│   │   ├── Module.hs       # Import graph
│   │   ├── Flake.hs        # Flake structure
│   │   └── Format.hs       # Type annotations
│   ├── Bash/
│   │   ├── Parse.hs        # ShellCheck wrapper
│   │   ├── Facts.hs        # Fact extraction
│   │   └── Patterns.hs     # Pattern matching
│   ├── Lint/
│   │   └── Forbidden.hs    # Heredoc/eval detection
│   └── Infer/
│       ├── Constraint.hs   # Constraint generation
│       └── Unify.hs        # Unification

src/armitage/
├── Armitage/
│   ├── DICE.hs             # Incremental computation
│   ├── Nix.hs              # Flake resolution
│   ├── RE.hs               # Remote execution
│   ├── CAS.hs              # Content-addressed storage
│   ├── Builder.hs          # Build orchestration
│   └── Store.hs            # Store operations
```

### New Work

| Component | Location | Status |
|-----------|----------|--------|
| `Render.Nix.Layout` | `nix/render/lib/` | **New** |
| `FlakeMetadata` type | `nix/render/lib/` | **New** |
| Render → Armitage bridge | `src/armitage/` | **New** |
| Layout validation | `nix/render/lib/` | **New** |
| Index generation | `nix/render/lib/` | **New** |

### Phase 1: Layout Validation

```bash
aleph check .
# Validates:
#   - No _index.nix files
#   - All modules have _class
#   - Directory structure matches kind
#   - No with/rec/IFD
#   - Parse succeeds
```

### Phase 2: Metadata Extraction

```bash
aleph check --emit-metadata .
# Outputs: FlakeMetadata (JSON or binary)
```

### Phase 3: Armitage Integration

```bash
aleph build .
# render → FlakeMetadata → armitage → DICE → store
```

### Phase 4: Daemon Bypass

```bash
aleph build --no-daemon .
# Writes directly to /nix/store with attestation
```

## Migration

### Existing Flakes

1. Run `aleph check .` to find violations
2. Fix `with` → `inherit (expr) name;`
3. Fix `rec` → `let` bindings or explicit args
4. Remove `_index.nix` files
5. Add `_class` to modules
6. Run `aleph check .` until clean

### Automated Fixes (Future)

```bash
aleph fix .              # Auto-fix where safe
aleph fix --dry-run .    # Show what would change
```

## FAQ

### Why ban `with`?

`with` introduces names into scope dynamically. Given:

```nix
with foo; { inherit bar; }
```

We cannot know statically whether `bar` comes from `foo` or an outer scope.
This breaks:
- Go-to-definition
- Type inference
- Import graph extraction
- Tooling in general

Use `inherit (foo) bar;` instead. Explicit is better than implicit.

### Why ban `rec`?

`rec` enables self-reference:

```nix
rec { x = x + 1; }  # Infinite loop
```

This breaks:
- Termination analysis
- Topological sorting
- Static evaluation

Use `let` bindings or explicit function arguments.

### What about existing nixpkgs code?

This RFC applies to **flakes using aleph**. Nixpkgs is not required to follow
these rules. When you depend on nixpkgs, armitage resolves those dependencies
via the flake input, not by analyzing nixpkgs source.

### Can I use aleph without all these restrictions?

No. The restrictions are the contract. Without them, static extraction is
impossible, and you must use the daemon.

### What's the performance benefit?

Without the daemon:
- No eval/build/sign round-trips
- Parallel execution via DICE
- Cache hits at action level, not derivation level
- Remote execution via NativeLink

Preliminary: 10-100x faster for incremental builds.

## References

- RFC-008: The Continuity Project
- RFC-011: render.nix (superseded)
- RFC-012: render.nix - Typed Shell Scripts (superseded)
- [Buck2 DICE](https://buck2.build/docs/concepts/dice/)
- [hnix](https://github.com/haskell-nix/hnix)
- [ShellCheck](https://www.shellcheck.net/)
- [NativeLink](https://github.com/TraceMachina/nativelink)
