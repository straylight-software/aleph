# Continuity Roadmap: Tightening the Noose

## The Strategy

We start with full Nix compatibility (Dhall → Nix derivations) and progressively
constrain the system until only typed, hermetic, content-addressed builds remain.

```
Phase 0: Dhall → Nix (full compat)
    ↓
Phase 1: Dhall → Nix (constrained derivations)
    ↓
Phase 2: Dhall → DICE + Nix (hybrid)
    ↓
Phase 3: Dhall → DICE (Nix for bootstrap only)
    ↓
Phase 4: DICE only (Nix eliminated)
```

---

## Phase 0: Full Nix Compatibility (Now)

**Goal:** Write derivations in Dhall, compile to Nix, use existing infrastructure.

### What works:
- `Derivation.dhall` → `dhall-to-nix` → `.nix` files
- All existing Nix tooling (nixpkgs, flakes, etc.)
- `nix build`, `nix develop`, etc.

### Escape hatches:
- Raw Nix expressions via `builtins.toNix`
- Arbitrary `env` variables
- Custom phases
- `allowedReferences = None` (unrestricted)

### Example:
```dhall
let D = ./prelude/Derivation.dhall

let myPackage = D.mkDerivation "hello" (D.Input.Src ./.)
    // { buildPhase = Some "make"
       , installPhase = Some "make install PREFIX=$out"
       }

in myPackage
```

---

## Phase 1: Constrained Derivations (Month 1-2)

**Goal:** Restrict what derivations can express. Start closing escape hatches.

### Constraints introduced:
```dhall
let ConstrainedDerivation =
      { -- REQUIRED: explicit system
        system : System
        -- REQUIRED: typed builder (not arbitrary path)
      , builder : Builder
        -- REQUIRED: all inputs declared
      , inputs : List Input
        -- BANNED: no raw env vars
        -- env : List EnvVar  -- REMOVED
        -- BANNED: no custom phases
        -- phases : List BuildPhase  -- REMOVED
      }
```

### What's banned:
- [ ] `env` with arbitrary strings
- [ ] Custom `buildPhase` as raw shell
- [ ] `allowSubstitutes = False` (must use cache)
- [ ] `__impure` anything

### What's required:
- [x] All inputs must be `Input.Drv` or `Input.Store` (no `Input.Src ./foo`)
- [x] All outputs must have declared names
- [x] `allowedReferences` must be `Some [...]` (explicit)

### Migration:
```bash
straylight lint --phase1  # Warns on phase0 escape hatches
straylight migrate --phase1  # Suggests fixes
```

---

## Phase 2: Hybrid DICE + Nix (Month 3-4)

**Goal:** New builds use DICE, existing Nix derivations wrapped.

### Architecture:
```
BUILD.dhall (rules)
    │
    ├──→ DICE actions (new code)
    │
    └──→ Nix derivations (wrapped legacy)
            │
            └──→ Eventually: converted to DICE
```

### The `NixCompat` wrapper:
```dhall
let Action = ./prelude/Action.dhall

-- Wrap a Nix derivation as a DICE action
let nixCompat =
      \(drv : Derivation) ->
        Action.Action
          { category = Action.ActionCategory.Custom "nix-compat"
          , identifier = drv.name
          , inputs = drvInputsToActionInputs drv.inputs
          , outputs = drvOutputsToActionOutputs drv.outputs
          , command = [ "nix", "build", "--no-link", drvToExpr drv ]
          , env = [] : List Action.EnvVar
          , toolchain = None Toolchain.Toolchain
          }
```

### What's new:
- DICE builds for Rust, Lean, C++ (our languages)
- Remote execution available
- Incremental builds

### What's legacy:
- Nix derivations for bootstrap compilers
- Nix derivations for complex packages (llvm, gcc, etc.)

---

## Phase 3: DICE Primary (Month 5-6)

**Goal:** DICE for all application code. Nix only for toolchain bootstrap.

### What uses DICE:
- All `rust_*` rules
- All `lean_*` rules
- All `cxx_*` rules
- All `haskell_*` rules
- All `purescript_*` rules
- All `nv_*` rules

### What uses Nix (bootstrap only):
- `rustc` itself
- `ghc` itself
- `clang` itself
- `lean` itself
- System libraries (glibc, etc.)

### The Bootstrap Boundary:
```dhall
-- Toolchains are still Nix-provided
let rustc = Toolchain.fromNix "nixpkgs#rustc"
let clang = Toolchain.fromNix "nixpkgs#clang"

-- But builds use DICE
let myLib = S.rust_library "mylib" [...] [...]
    // { toolchain = Some rustc }
```

### Constraints tightened:
- [ ] No `Derivation.dhall` in application code
- [ ] No `mkDerivation` (use language-specific rules)
- [ ] No raw shell in build phases

---

## Phase 4: DICE Only (Month 7+)

**Goal:** Nix eliminated. DICE builds everything, including compilers.

### What changes:
- Toolchains are DICE-built (from source or content-addressed binaries)
- No Nix evaluator in the build path
- R2 + git replaces Nix store

### The Final Stack:
```
BUILD.dhall
    │
    ▼ dhall eval
    │
    ▼ DICE actions
    │
    ├──→ Remote execution (optional)
    │
    ▼ Artifacts
    │
    ▼ R2 (bytes) + git (attestation)
```

### What's gone:
- [x] Nix evaluator
- [x] Nix store (replaced by R2)
- [x] Nix daemon
- [x] `.drv` files (replaced by DICE action keys)
- [x] NAR format (replaced by content-addressed blobs)

### What remains:
- Content addressing (same concept, different implementation)
- Hermeticity (enforced by DICE + namespace/VM isolation)
- Reproducibility (proven by coset equivalence)

---

## The Noose Tightening Schedule

| Phase | Escape Hatches | Timeline |
|-------|----------------|----------|
| 0 | Everything allowed | Now |
| 1 | No raw env, no custom phases | Month 2 |
| 2 | Nix only for wrapped legacy | Month 4 |
| 3 | Nix only for bootstrap | Month 6 |
| 4 | No Nix | Month 7+ |

## Lint Levels

```bash
# Phase 0: Anything goes
straylight lint --level=permissive

# Phase 1: Warn on escape hatches
straylight lint --level=constrained

# Phase 2: Error on non-DICE builds (except wrapped)
straylight lint --level=dice-preferred

# Phase 3: Error on any Nix in application code
straylight lint --level=dice-required

# Phase 4: Error on any Nix anywhere
straylight lint --level=pure-dice
```

## Migration Commands

```bash
# Show what needs to change for next phase
straylight migrate --to=phase1 --dry-run

# Auto-fix what can be fixed
straylight migrate --to=phase1 --fix

# Convert a Nix derivation to DICE
straylight convert --from=nix --to=dice ./default.nix

# Wrap a Nix derivation as DICE (temporary)
straylight wrap --nix ./default.nix > BUILD.dhall
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] All escape hatches produce warnings
- [ ] 80% of builds pass `--level=constrained`

### Phase 2 Complete When:
- [ ] All new code uses DICE
- [ ] Nix wrapper exists for legacy
- [ ] Remote execution works

### Phase 3 Complete When:
- [ ] No `Derivation.dhall` in application code
- [ ] Nix only in `toolchains/` directory
- [ ] Build times improved (incremental DICE)

### Phase 4 Complete When:
- [ ] `straylight build` works without Nix installed
- [ ] All artifacts in R2
- [ ] All attestations in git
- [ ] Nix is a historical curiosity
