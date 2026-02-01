# Continuity Roadmap: Tightening the Noose

**Last Updated:** 2026-01-27

## Current Status

We are in **Phase 1.5** - infrastructure is built, integration is next.

### What's Built

| Component | LOC | Status |
|-----------|-----|--------|
| Armitage (Haskell) | 7.7k | Working - CAS, RE, DICE core |
| DICE reference (Rust) | 36k | Working - benchmarks run |
| Buck2 prelude | 5.6k | Working - Haskell/Rust toolchains |
| NativeLink integration | - | Working - gRPC client complete |
| tvix-eval | 16k | Evaluating - nixpkgs-compat |

### What's Next

1. Wire tvix-eval or nix-compat for derivation hashing
2. Complete DICE ↔ NativeLink action execution
3. Run first nixpkgs derivation through armitage

## The Strategy

We start with full Nix compatibility (Dhall → Nix derivations) and progressively
constrain the system until only typed, hermetic, content-addressed builds remain.

```
Phase 0: Dhall → Nix (full compat)
    ↓
Phase 1: Dhall → Nix (constrained derivations)    ← INFRASTRUCTURE BUILT
    ↓
Phase 1.5: Nix → armitage → NativeLink            ← WE ARE HERE
    ↓
Phase 2: Dhall → DICE + Nix (hybrid)
    ↓
Phase 3: Dhall → DICE (Nix for bootstrap only)
    ↓
Phase 4: DICE only (Nix eliminated)
```

---

## Phase 0: Full Nix Compatibility (Complete)

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

## Phase 1.5: Infrastructure Complete (Current)

**Goal:** All infrastructure built, integration in progress.

### What's Built

#### Armitage (7.7k LOC Haskell)
```
src/armitage/
├── Armitage/DICE.hs       975 LOC  ✓ Incremental computation core
├── Armitage/Proto.hs      910 LOC  ✓ Hand-rolled RE protobuf (no codegen)
├── Armitage/Proxy.hs      890 LOC  ✓ Witness proxy for fetches
├── Armitage/Dhall.hs      742 LOC  ✓ BUILD.dhall evaluation
├── Armitage/RE.hs         663 LOC  ✓ gRPC remote execution client
├── Armitage/Builder.hs    583 LOC  ✓ Build orchestration
├── Armitage/Trace.hs      518 LOC  ✓ Execution traces (CBOR)
├── Armitage/Nix.hs        499 LOC  ✓ Nix derivation parsing
├── Armitage/CAS.hs        380 LOC  ✓ Content-addressed storage client
├── Armitage/Store.hs      348 LOC  ✓ Artifact store abstraction
└── Armitage/Toolchain.hs  317 LOC  ✓ Toolchain management
```

#### DICE Reference (36k LOC Rust - Meta's implementation)
```
src/dice-rs/
├── dice/                 19.2k LOC  ✓ Core engine (ported from Buck2)
├── allocative/            4.1k LOC  ✓ Memory profiling
├── dice_futures/          2.4k LOC  ✓ Async/cancellation
├── bench/                   308 LOC  ✓ Benchmarks (running)
└── ...support crates

Benchmark results:
  cache_hits:     ~91ns/iter
  linear_chain:   ~195ns/iter (10 nodes)
  incremental:    ~8.9ms/iter (1000 nodes)
```

#### Buck2 Prelude (5.6k LOC Starlark)
```
prelude/
├── haskell/              ✓ GHC toolchain, libraries, binaries
├── cxx/                  ✓ C++ compilation, linking
├── linking/              ✓ Shared/static libraries
├── platforms/            ✓ Platform constraints
└── 57 modules total
```

#### NativeLink Integration
- [x] gRPC CAS client (FindMissingBlobs, BatchUpdateBlobs, BatchReadBlobs)
- [x] gRPC Execute client (Execute, WaitExecution)
- [x] Directory serialization (recursive tree upload)
- [x] Fly.io deployment working

### What's Missing (Next Steps)

1. **Nix Language Evaluation**
   - Option A: FFI to tvix-eval (~16k LOC Rust)
   - Option B: Port tvix nix-compat (~2k LOC) for derivation hashing only
   - Decision: Evaluate tvix - it's nixpkgs-compatible, not bug-for-bug

2. **DICE ↔ Action Execution Bridge**
   - Wire Haskell DICE to call NativeLink Execute
   - Map DICE keys to RE action digests
   - Handle action cache lookups

3. **First nixpkgs Build**
   - Parse `.drv` file (tvix nix-compat or port)
   - Compute `hashDerivationModulo`
   - Execute via armitage → NativeLink
   - Verify output hash matches

### tvix Evaluation

tvix is the clean Nix implementation from TVL. Key facts:

| Aspect | C++ Nix | tvix |
|--------|---------|------|
| LOC (eval) | ~100k | ~16k |
| Store coupling | Tight | Separate |
| Bug compat | All bugs | nixpkgs only |
| Language | C++ | Rust |

**Decision**: Use tvix-eval temporarily, port nix-compat long-term.

The only part we need permanently is `hashDerivationModulo` (~50 LOC).
Everything else is temporary until Dhall replaces Nix lang.

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

| Phase | Escape Hatches | Timeline | Status |
|-------|----------------|----------|--------|
| 0 | Everything allowed | - | Complete |
| 1 | No raw env, no custom phases | Month 2 | Complete |
| 1.5 | Infrastructure built | Month 3 | **Current** |
| 2 | Nix only for wrapped legacy | Month 4 | Next |
| 3 | Nix only for bootstrap | Month 6 | Planned |
| 4 | No Nix | Month 7+ | Planned |

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
- [x] All escape hatches produce warnings
- [x] 80% of builds pass `--level=constrained`

### Phase 1.5 Complete When:
- [x] Armitage CAS client working (gRPC to NativeLink)
- [x] Armitage RE client working (Execute API)
- [x] DICE reference implementation running (Rust benchmarks)
- [x] Buck2 prelude extracted and working
- [ ] tvix-eval or nix-compat integrated
- [ ] First nixpkgs derivation built via armitage

### Phase 2 Complete When:
- [ ] All new code uses DICE
- [ ] Nix wrapper exists for legacy
- [ ] Remote execution works end-to-end

### Phase 3 Complete When:
- [ ] No `Derivation.dhall` in application code
- [ ] Nix only in `toolchains/` directory
- [ ] Build times improved (incremental DICE)

### Phase 4 Complete When:
- [ ] `straylight build` works without Nix installed
- [ ] All artifacts in R2
- [ ] All attestations in git
- [ ] Nix is a historical curiosity

---

## Complexity Budget

The goal is radical simplification. Current vs. target:

| Component | C++ Nix / Buck2 | Straylight Target |
|-----------|-----------------|-------------------|
| Language evaluator | 100k LOC | 0 (Dhall is total) |
| Build orchestration | 50k LOC | ~2k LOC (DICE core) |
| Execution layer | 50k LOC | ~500 LOC (derivation = action) |
| Store | 20k LOC | ~500 LOC (R2 is CAS) |
| Total | ~220k LOC | ~3k LOC |

Why the compression?
1. **Dhall is total** - no eval complexity, no thunks, no laziness
2. **Derivation is action** - no action type zoo
3. **R2 is CAS** - no store implementation
4. **Content-addressing is structural** - no hashing step
5. **Typed triples** - no platform sniffing

The 220k → 3k compression (~99%) comes from eliminating unsoundness.
