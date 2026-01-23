# Dhall-to-Buck2 Bridge Options

## The Problem

We need to evaluate Dhall BUILD files and convert them to Buck2 action graphs.
There are several approaches with different tradeoffs.

## Option 1: dhall-rust → WASM → builtins.wasm (High Effort, High Reward)

**Path:** Dhall → dhall-rust (WASM) → Nix via builtins.wasm

This is the "write Nix derivations in Dhall" path.

### Status of dhall-rust
- **Last commit:** 2025-09-10 (active enough)
- **Lines of code:** ~8,229 in core
- **WASM support:** Partial - `cfg(not(target_arch = "wasm32"))` for reqwest
- **Builds without reqwest:** YES

### What's needed:
1. Fork dhall-rust
2. Add WASM target (disable reqwest, use local imports only)
3. Expose `evaluate(dhall_source) -> json/cbor` ABI
4. Integrate with existing builtins.wasm infrastructure

### Pros:
- Dhall → Nix derivations during transition
- Reuses existing straylight-nix infrastructure
- Type-safe Nix expressions

### Cons:
- Still going through Nix evaluator
- ~8K lines to maintain
- Dhall semantics frozen (can't extend easily)

### Effort: **Medium** (2-4 weeks)

---

## Option 2: dhall-rust → JSON → Buck2 BXL (Medium Effort)

**Path:** Dhall → dhall-rust CLI → JSON → Buck2 BXL script

### What's needed:
1. Use dhall-rust as-is (it exports JSON already)
2. Write BXL script to consume JSON and create targets
3. `straylight build` shells out to dhall → BXL

### Pros:
- No modifications to dhall-rust needed
- Uses existing Buck2 machinery
- Quick to prototype

### Cons:
- Shell-out overhead
- BXL is still Starlark (ironic)
- Two-phase evaluation

### Effort: **Low** (1 week)

---

## Option 3: dhall-haskell → dhall-nix → Nix (Existing)

**Path:** Dhall → dhall-to-nix → Nix expressions → nix-build

This already exists in dhall-haskell.

### What's needed:
1. Package dhall-to-nix
2. Write Nix library to interpret dhall-generated attrsets
3. Bridge to existing prelude

### Pros:
- Already exists and maintained
- Proven implementation
- GHC WASM can compile it (future)

### Cons:
- Haskell dependency (large)
- Not integrated with Buck2
- Still Nix-centric

### Effort: **Low** (1 week for integration)

---

## Option 4: Native Rust Buck2 Integration (High Effort, Best Result)

**Path:** Dhall → dhall-rust (native) → Buck2 Rust API

Buck2 is Rust. dhall-rust is Rust. Why shell out?

### What's needed:
1. Fork dhall-rust into straylight repo
2. Add `dhall` crate as Buck2 dependency
3. Implement `DhallConfigurationContext` in Buck2
4. Replace Starlark BUCK parsing with Dhall

### The key insight:
Buck2's `prelude/` is ~60K lines of Starlark.
If we replace it with Dhall types, we get:
- Type checking at parse time
- No Starlark interpreter overhead
- Content-addressed imports (Dhall's killer feature)

### Architecture:
```
BUILD.dhall
    │
    ▼ dhall-rust (embedded in buck2)
    │
    ▼ Dhall Value → Buck2 TargetNode
    │
    ▼ Buck2 action graph
    │
    ▼ DICE execution
```

### What Buck2 needs from Dhall:
```rust
// In buck2
use dhall::{Parsed, Ctxt};

fn load_build_file(path: &Path) -> Result<Vec<TargetNode>> {
    let cx = Ctxt::new();
    let parsed = Parsed::parse_file(path)?;
    let resolved = parsed.resolve(cx)?;
    let typed = resolved.typecheck(cx)?;
    let normalized = typed.normalize(cx);
    
    // Convert Dhall value to Buck2 targets
    dhall_to_targets(normalized)
}
```

### Pros:
- Native integration, no shell-out
- Type errors at parse time
- Eliminates Starlark entirely
- Content-addressed imports built-in
- One Rust binary

### Cons:
- Fork of Buck2 diverges from upstream
- ~8K lines of dhall-rust to maintain
- Dhall semantics must match Buck2 expectations

### Effort: **High** (4-8 weeks)

---

## Option 5: Dhall → Starlark Transpiler (Actually Good)

**Path:** Dhall → transpile to Starlark BUCK → existing Buck2

This is less cursed than it sounds. Key insight: BUCK files are **data**, not programs.

### What a BUCK file actually is:
```python
# This is just function calls with keyword args
rust_library(
    name = "mylib",
    srcs = ["src/lib.rs"],
    deps = [":other"],
)
```

### What Dhall produces:
```dhall
let DICE = ./package.dhall

in [ DICE.rust_library "mylib" ["src/lib.rs"] [":other"] ]
```

### The transpilation is trivial:
```
Dhall Record → Starlark function call
Dhall List → Starlark list
Dhall Text → Starlark string
Dhall Union → (depends on schema)
```

### What's needed:
1. dhall-rust evaluates to normalized value
2. Small Rust printer: `Value → String` (Starlark syntax)
3. Write to `.buck` or pipe to Buck2

### Example transpiler (~100 lines):
```rust
fn to_starlark(value: &Nir) -> String {
    match value.kind() {
        NirKind::RecordLit(fields) => {
            // { name = "x", srcs = [...] } → name = "x", srcs = [...]
            fields.iter()
                .map(|(k, v)| format!("{} = {}", k, to_starlark(v)))
                .collect::<Vec<_>>()
                .join(", ")
        }
        NirKind::List(items) => {
            format!("[{}]", items.iter()
                .map(to_starlark)
                .collect::<Vec<_>>()
                .join(", "))
        }
        NirKind::Text(s) => format!("\"{}\"", s.escape_default()),
        NirKind::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        NirKind::Natural(n) => n.to_string(),
        // ... etc
    }
}

fn emit_target(rule_name: &str, attrs: &Nir) -> String {
    format!("{}({})", rule_name, to_starlark(attrs))
}
```

### Architecture:
```
BUILD.dhall
    │
    ▼ dhall-rust (evaluate)
    │
    ▼ Normalized Dhall Value
    │
    ▼ to_starlark() printer (~100 LOC)
    │
    ▼ BUCK file (text)
    │
    ▼ Buck2 (unmodified)
```

### Pros:
- Works with **unmodified Buck2**
- Trivial transpilation (100 lines)
- Type checking happens in Dhall (before Buck2 sees it)
- Can use existing prelude unchanged
- Incremental: migrate one package at a time
- Can run `dhall format`, `dhall lint`, etc.

### Cons:
- Generated BUCK files are "build artifacts"
- Two-phase: dhall eval → buck2 build
- Errors reference generated code (but dhall errors are better anyway)

### Why this is actually good:
1. **Dhall catches errors first** - type errors, missing fields, wrong types
2. **Buck2 sees valid Starlark** - no Buck2 modifications needed
3. **Existing tooling works** - `buck2 query`, `buck2 audit`, etc.
4. **Migration is trivial** - run transpiler, commit BUCK, delete BUILD.dhall later

### Effort: **Low** (1-2 days for transpiler)

---

## Recommendation

### For Immediate Use (Now): Option 5

**Dhall → Starlark transpiler** is the clear winner for bootstrapping:
- 1-2 days to implement
- Works with unmodified Buck2
- Type safety in Dhall, execution in Buck2
- Can migrate incrementally

### For Transition (1-6 months): Option 5 + 3

Keep using **Dhall → Starlark** for Buck2.
Add **dhall-to-nix** for Nix derivations during transition.

### For Final State (6+ months): Option 4

Fork Buck2, embed dhall-rust, eliminate Starlark entirely.

This gives us:
- Single Rust binary (straylight/diced)
- Type checking at parse time
- Content-addressed everything
- No interpreter overhead

But Option 5 is good enough that we might never need Option 4.

---

## Implementation Plan

### Phase 1: Transpiler (Days 1-2)
1. Add dhall-rust to aleph (already builds without reqwest)
2. Write `to_starlark()` printer (~100 LOC)
3. Wire up: `straylight gen //path:BUILD.dhall` → `BUCK`
4. Build one real target

### Phase 2: Dhall Schemas (Week 1)
1. Port Buck2 `decls/*.bzl` to Dhall types
2. `rust_library`, `rust_binary`, `c_library`, etc.
3. Toolchain types
4. Test against real prelude

### Phase 3: Migration (Week 2-4)
1. Convert existing BUCK files to BUILD.dhall
2. Integrate with `straylight build` (auto-transpile)
3. Add dhall-to-nix for Nix compat
4. CI: check that generated BUCK matches committed

### Phase 4: Polish (Week 4+)
1. LSP integration (dhall-lsp-server)
2. Error message mapping
3. `straylight srcs` helper (list files, no globs)
4. Documentation

### Phase 5: Native Integration (Optional, 6+ months)
1. Fork Buck2
2. Embed dhall-rust
3. Replace Starlark parsing
4. Only if transpiler becomes a bottleneck

---

## Open Questions

1. **Dhall imports:** Should we allow remote imports or only local?
   - Local only for hermeticity
   - Content-addressed hash is the version

2. **Type evolution:** How do we evolve BUILD.dhall schemas?
   - Dhall has union types, can add variants
   - Breaking changes need migration

3. **Performance:** Is Dhall evaluation fast enough?
   - dhall-rust is reasonably fast
   - Caching normalized expressions helps

4. **Tooling:** LSP, formatting, etc?
   - dhall-lsp-server exists (Haskell)
   - Could port to Rust eventually
