# aleph-008: The Continuity Project

| Field | Value |
|-------|-------|
| RFC | aleph-008 |
| Title | The Continuity Project |
| Author | b7r6 |
| Status | Draft |
| Created | 2026-01-22 |

## Abstract

Continuity is continuity. Continuity's job is continuity.

All computations run on "perfect conceptual computers." Correct by construction -
the result is saved. One content-addressing scheme - the hash is the artifact.
CA-derivations, Buck2, and Bazel are supports for a coset - they produce the
same cache keys because they're all content-addressing.

This RFC describes the unified build/cache/attestation system called
**straylight**, which eliminates redundancy in favor of superior alternatives
and composes a minimal set of atoms into a complete computational universe.

## Core Thesis

### The Atoms

Plan 9 failed because "everything is a file" is too simple. The algebra is
slightly bigger:

| Atom | What It Is | Implementation |
|------|------------|----------------|
| **namespace** | isolation boundary | `unshare` |
| **microvm** | compute unit | firecracker |
| **build** | computation with result | derivation |
| **store** | content-addressed storage | r2 |
| **identity** | cryptographic identity | ed25519 |
| **attestation** | signature on artifact | git |

These are the primitives. Not "hash -> bytes" - there's structure.

### Eliminations

Redundant things eliminated in favor of superior alternatives:

| Eliminated | Replaced By | Rationale |
|------------|-------------|-----------|
| Bazel | Buck2 (DICE) | Obsolete junk |
| Starlark | Dhall | System Fomega, total, sufficient |
| Preludes | Explicit typed toolchains | No magic |
| Globs | Explicit file lists | "I'll know what I'm building or I'll know the reason why" |
| String-typed configs | Real triples, real compilers, real flags | Type safety |
| nix remote builders | isospin | Build hook is a Superfund site |
| nix daemon | armitage | Witness proxy, coeffect-aware |
| "purity" boolean | coeffect algebra | Graded resources, not pure/impure |

### The Stack

```
straylight (CLI)
    │
    ├── armitage (nix shim)
    │       │
    │       ├── witness proxy (fetches → R2)
    │       ├── coeffect checker (Lean4)
    │       └── attestation (ed25519 + git)
    │
    ├── DICE (real builds)
    │       │
    │       ├── Dhall (BUILD files, System Fω, total)
    │       ├── Buck2 core (action graph, RE, TSET)
    │       └── Executor (wasm sandbox / isospin)
    │
    └── Store
            │
            ├── R2 (bytes, content-addressed)
            ├── git (attestations, refs)
            └── ed25519 (identity, signing)
```

See [armitage.md](armitage.md) for the nix compatibility layer design.

## Component 1: Dhall Build Language

Dhall is sufficient. It is:

- System Fomega (typed, higher-kinded)
- Total (guaranteed termination)
- Content-addressed imports (the hash IS the version)
- Hermetic (no IO, no escape)
- Simple (not a general purpose language)
- Exists (dhall-rust is lootable)

### BUILD.dhall

```dhall
-- BUILD.dhall

let DICE    = https://straylight.cx/dice/v1.dhall sha256:abc...
let Targets = https://straylight.cx/targets/v1.dhall sha256:def...
let Rust    = https://straylight.cx/rules/rust/v1.dhall sha256:123...
let Lean    = https://straylight.cx/rules/lean/v1.dhall sha256:456...
let C       = https://straylight.cx/rules/c/v1.dhall sha256:789...
let Wasm    = https://straylight.cx/rules/wasm/v1.dhall sha256:aaa...

-- targets
let targets =
    { native   = Targets.host
    , orin     = Targets.aarch64-linux { cpu = "cortex-a78ae" }
    , wasm     = Targets.wasm32-wasi
    }

-- proven core (lean -> c -> wasm, never changes)
let r2_backend = Lean.library
    { name = "r2-backend"
    , srcs = [ "core/r2_backend.lean" ]
    , deps = [ "//core:sha256", "//core:s3" ]
    , extract = C.extract
        { target = targets.wasm
        , verified = True
        }
    }

let git_odb = Lean.library
    { name = "git-odb"
    , srcs = [ "core/git_odb.lean" ]
    , deps = [ r2_backend ]
    , extract = C.extract
        { target = targets.wasm
        , verified = True
        }
    }

-- wasm modules from proven C
let builtins_wasm = Wasm.module
    { name = "builtins"
    , src = builtins
    , optimize = Wasm.O3
    , features = [ Wasm.bulk_memory, Wasm.simd ]
    }

-- rust tooling (uses proven wasm via wasmtime)
let straylight_core = Rust.library
    { name = "straylight-core"
    , srcs =
        [ "src/core/lib.rs"
        , "src/core/store.rs"
        , "src/core/artifact.rs"
        ]
    , deps =
        [ "//vendor:wasmtime"
        , "//vendor:git2"
        , "//vendor:aws-sdk-s3"
        , "//vendor:ed25519-dalek"
        ]
    , embed = [ builtins_wasm, git_odb ]
    , edition = Rust.Edition.2024
    }

let straylight_cli = Rust.binary
    { name = "straylight"
    , main = "src/main.rs"
    , deps = [ straylight_core ]
    , targets = [ targets.native, targets.orin ]
    }

in { straylight_cli, straylight_core, builtins_wasm }
```

### No Globs

```dhall
-- NO
srcs = glob "src/**/*.rs"

-- YES
srcs =
    [ "src/lib.rs"
    , "src/core/mod.rs"
    , "src/core/store.rs"
    , "src/core/artifact.rs"
    ]
```

The CLI proposes, human approves:

```bash
straylight srcs //mylib

# Found 3 new files not in BUILD.dhall:
#   + src/core/namespace.rs
#   + src/core/vm.rs
#   + src/transport/git.rs
#
# Add them? [y/n/edit]
```

Every file in the build is a choice, not an accident.

## Component 2: Typed Toolchains

Toolchains look complicated in Starlark. They aren't.

```
compiler + target + flags = toolchain
toolchain + sources + deps = build
build -> artifact -> hash

three lines.
```

### Toolchain Types

```dhall
-- https://straylight.cx/toolchains/v1.dhall

let Triple =
    { arch   : Arch
    , vendor : Vendor
    , os     : OS
    , abi    : ABI
    }

let Compiler =
    < Clang : { version : Version, artifact : Artifact }
    | Rustc : { version : Version, artifact : Artifact }
    | GHC   : { version : Version, artifact : Artifact }
    | Lean  : { version : Version, artifact : Artifact }
    >

let Toolchain =
    { compiler : Compiler
    , host     : Triple
    , target   : Triple
    , flags    : List Flag
    , linker   : Optional Artifact
    , sysroot  : Optional Artifact
    }
```

### Concrete Toolchains

```dhall
let clang_18 = Compiler.Clang
    { version = v "18.1.0"
    , artifact = sha256:abc...
    }

let rust_1_80 = Compiler.Rustc
    { version = v "1.80.0"
    , artifact = sha256:def...
    }

let orin_sysroot = sha256:deadbeef...

let toolchains =
    { native_rust =
        { compiler = rust_1_80
        , host     = triples.x86_64_linux
        , target   = triples.x86_64_linux
        , flags    = [ Flag.TargetCpu Cpu.native ]
        , linker   = None Artifact
        , sysroot  = None Artifact
        }

    , orin_rust =
        { compiler = rust_1_80
        , host     = triples.x86_64_linux
        , target   = triples.aarch64_linux
        , flags    = [ Flag.TargetCpu Cpu.cortex_a78ae ]
        , linker   = Some clang_18
        , sysroot  = Some orin_sysroot
        }

    , wasm_rust =
        { compiler = rust_1_80
        , host     = triples.x86_64_linux
        , target   = triples.wasm32_wasi
        , flags    = [ Flag.OptLevel OptLevel.Oz ]
        , linker   = None Artifact
        , sysroot  = None Artifact
        }
    }
```

### Typed Flags

```dhall
let Flag =
    < OptLevel     : OptLevel
    | TargetCpu    : Cpu
    | LTO          : LTOMode
    | Codegen      : CodegenFlag
    | Link         : LinkFlag
    | Debug        : DebugInfo
    | Feature      : { enable : Bool, name : Text }
    | Raw          : Text  -- escape hatch, logged + warned
    >

let LTOMode = < Off | Thin | Fat >
```

Flags compose, validate, dedupe. Not strings.

## Component 3: Target Types

```dhall
-- https://straylight.cx/targets/v1.dhall

let Arch = < x86_64 | aarch64 | wasm32 | riscv64 >

let OS = < linux | darwin | wasi | none >

let Cpu = < generic
          | cortex-a78ae    -- orin
          | neoverse-n1     -- graviton
          | znver3          -- zen3
          >

let Target =
    { arch   : Arch
    , os     : OS
    , cpu    : Cpu
    , triple : Text  -- derived, but explicit
    }

let aarch64-linux =
    \(cfg : { cpu : Cpu }) ->
    { arch = Arch.aarch64
    , os = OS.linux
    , cpu = cfg.cpu
    , triple = "aarch64-unknown-linux-gnu"
    } : Target

let wasm32-wasi =
    { arch = Arch.wasm32
    , os = OS.wasi
    , cpu = Cpu.generic
    , triple = "wasm32-wasi"
    } : Target
```

## Component 4: Rust Object Model

Preludeless Rust in idiomatic Dhall. Useful for code generation, FFI bindings,
AST representation, and the stochastic_omega translation pipeline.

```dhall
-- https://straylight.cx/lang/rust/v1.dhall

let Primitive =
    < U8 | U16 | U32 | U64 | U128 | Usize
    | I8 | I16 | I32 | I64 | I128 | Isize
    | F32 | F64
    | Bool
    | Char
    | Str
    | Never
    >

let Mutability = < Mut | Const >

let Lifetime = < Static | Named : Text | Elided >

let Type =
    < Prim     : Primitive
    | Ref      : { lifetime : Lifetime, mut : Mutability, inner : Type }
    | Ptr      : { mut : Mutability, inner : Type }
    | Array    : { inner : Type, len : Natural }
    | Slice    : { inner : Type }
    | Tuple    : List Type
    | Fn       : { args : List Type, ret : Type }
    | Path     : { segments : List Text, generics : List Type }
    | Generic  : Text
    | Unit
    >

let Struct =
    < Unit   : { name : Text }
    | Tuple  : { name : Text, fields : List Type }
    | Fields : { name : Text, fields : List Field }
    >

let Variant =
    < Unit   : Text
    | Tuple  : { name : Text, fields : List Type }
    | Struct : { name : Text, fields : List Field }
    >

let Enum =
    { name     : Text
    , generics : List Generic
    , variants : List Variant
    }

let Fn =
    { name       : Text
    , generics   : List Generic
    , params     : List Param
    , ret        : Type
    , vis        : Visibility
    , unsafe     : Bool
    , const      : Bool
    , async      : Bool
    , extern_abi : Optional ABI
    }

let Trait =
    { name       : Text
    , generics   : List Generic
    , supertraits: List TraitRef
    , types      : List AssocType
    , methods    : List TraitMethod
    }

let Impl =
    < Inherent :
        { generics : List Generic
        , self_ty  : Type
        , items    : List ImplItem
        }
    | Trait :
        { generics : List Generic
        , trait_   : TraitRef
        , self_ty  : Type
        , items    : List ImplItem
        }
    >

let Crate =
    { name  : Text
    , items : List Item
    }
```

### Usage: Define Option Without Prelude

```dhall
let Rust = https://straylight.cx/lang/rust/v1.dhall sha256:...

let T = Rust.Generic { name = "T", bounds = [] : List Rust.Bound, default = None Rust.Type }

let Option = Rust.Enum
    { name = "Option"
    , generics = [ T ]
    , variants =
        [ Rust.Variant.Unit "None"
        , Rust.Variant.Tuple { name = "Some", fields = [ Rust.Type.Generic "T" ] }
        ]
    }
```

Dhall IS the typed AST. The printer is one page.

## Component 5: Storage Architecture

### R2 is the Big Store in the Sky

Von Neumann's style - it won by economic necessity. R2 is:

- Cheap
- Free egress
- S3-compatible
- The obvious choice

### Git for Attestation

Git can talk to the big store in the sky:

- Git stores attestation (small)
- R2 stores artifacts (big)
- Git references point to R2 objects
- ed25519 for signing
- ssh for transport

### libgit2 ODB Backend

The libgit2 ODB backend for R2 is a one-shot if you can write it in
System Fomega or CIC. Correct by construction, write once, never touch again.

```
git objects -> sha256 -> r2 keys
git refs -> r2 metadata
git pack -> r2 multipart
```

## Component 6: stochastic_omega

A Lean4 tactic. LLM-driven proof search constrained by `rfl` (reflexivity).

```
1. generate Dhall (typed, total)
2. typecheck (dhall)
3. emit Rust source (trivial printer)
4. compile (rustc)
5. test
6. if property fails -> refine Dhall
```

The type system is the oracle that accepts or rejects. Use probabilistic search
(LLM sampling) through the space of Fomega terms.

- "stochastic" = the droid's search is probabilistic/ML-driven
- "omega" = System Fomega, the type operator level of the lambda cube

**stochastic_omega is a Lean4 tactic. System Omega* is what it can do.**

### The Degenerate Case

Lean compiles to certified C. The full pipeline:

1. stochastic_omega generates verified code
2. Lean4 compiles to C (certified extraction)
3. C compiles to native/wasm
4. The artifact is content-addressed

Proofs are erased after compilation - they were checked at compile time.

### Roundtripping Through rfl

Put a droid in a loop trying to roundtrip PureScript's prelude through
rfl-carrying ASTs. If the AST carries proofs that transformations are
identity-preserving, and you iterate with LLM proposals + rfl verification,
you converge on proven transformations.

## Component 7: The Language Coset

Same semantics, different tradeoffs:

```
PureScript <---> Haskell <---> Rust <---> Lean
     |              |            |          |
     v              v            v          v
    bun           GHC         rustc       lake
  (fast)        (native)     (native)      (C)

same logic. property tested against each other.
```

### How Close is PureScript to Rust?

At the LLVM IR level, closer than you'd think. If both compile down to similar
LLVM IR, the high-level language differences become irrelevant for the artifact.

### syn in Lean4

Can syn (Rust parser) be translated to Lean4? If syn existed in Lean4:

- Parse Rust source
- Verified parsing
- Round-trip through the proof system
- The Rust parser itself becomes part of the rfl nexus

This closes the loop.

## Component 8: isospin microvm

A minimal VM for GPU workloads with a proven driver stack.

### nvidia.ko is in-tree

NVIDIA open-sourced their kernel modules. nvidia.ko is now in the Linux kernel
tree (GPL dual licensed). This changes everything.

### Hoisting the Driver to Lean

If you hoist the nvidia driver source into Lean:

1. Minimal VM with just enough for GPU workloads
2. Proven driver (Lean -> C)
3. Minimal kernel surface
4. True isolation (hypervisor, not namespaces)

## The Straylight Cube

```
                         THE STRAYLIGHT CUBE

                          PROVEN (rfl)
                              |
                              |
                 Lean --------+-------- Liquid Haskell
                   |          |              |
                   |          |              |
           TOTAL --+----------+------------- +-- ESCAPE
                   |          |              |
               Dhall          |          PureScript
                   |          |              |
                   |          |          Haskell
                   |          |              |
                   +----------+------------- +--- Rust
                              |              |
                              |              |
                        stochastic_omega
                         (rfl oracle)
                              |
                 +------------+------------+
                 |                         |
                 v                         v
         WASM (portable)           C (native/kernel)
                 |                         |
                 |                         |
 +---------------+--------------------------+---------------+
 |               |                         |               |
 |               v                         v               |
 |         +---------+            +-------------+          |
 |         |builtins |            |   isospin   |          |
 |         |  .wasm  |            |  (microvm)  |          |
 |         +----+----+            +------+------+          |
 |              |                        |                 |
 |    ISOLATION |                        | ISOLATION       |
 |    (sandbox) |                        | (hypervisor)    |
 |              |                        |                 |
 |              |                        +-- nvidia.ko     |
 |              |                        |   (proven)      |
 +--------------|------------------------|-----------------+
                |                        |
                +-----------+------------+
                            |
                            v
                     +------------+
                     |    DICE    |
                     |            |
                     |  inputs    |
                     |     |      |
                     |  action    |
                     |     |      |
                     |  outputs   |
                     +-----+------+
                           |
                           v
       +---------------------------------------+
       |              ATOMS                    |
       |                                       |
       | namespace -- vm -- build -- artifact  |
       |      |       |       |         |      |
       |      +-------+-------+---------+      |
       |                  |                    |
       |               store                   |
       +------------------+--------------------+
                          |
             +------------+------------+
             |            |            |
             v            v            v
         +------+    +--------+   +---------+
         |sha256|    |  git   |   |   r2    |
         |      |    |        |   |         |
         | hash |<---|attest  |-->|  bytes  |
         +------+    +--------+   +---------+
                          |
                          |
                     +----+----+
                     | ed25519 |
                     |         |
                     |identity |
                     +----+----+
                          |
                          v
                     +---------+
                     |   ssh   |
                     |         |
                     |transport|
                     +---------+
```

## CLI Design

```bash
# nix compat mode (training wheels)
straylight run nixpkgs#hello
straylight shell nixpkgs#rust
    |
    +-- translates to nix, uses nix store (for now)

# native mode
straylight build //mylib
straylight build //mylib:wasm
straylight test //mylib:test

# the schema nix run nixpkgs#program is inadequate
# (we will support it as a special case)
```

## What's Left of Nix?

If we have:

1. PureNix for description (typed Fomega that compiles to Nix expressions)
2. builtins.wasm for evaluation (portable Nix evaluator)
3. wasm/firecracker for build isolation
4. git for content addressing
5. ed25519 for signing
6. ssh for transport

What's left of "Nix" as we know it?

- The DSL -> being replaced by typed languages
- The evaluator -> being replaced by builtins.wasm
- The daemon/store -> being replaced by r2 + git
- The sandbox -> being replaced by wasm/firecracker

**The answer: the protocol. The content-addressing scheme. The concept.**

## Implementation Priorities

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | CA derivations always-on | Complete (RFC-007) |
| 2 | builtins.wasm | Complete (RFC-007) |
| 3 | Typed package DSL | Complete (RFC-007) |
| 4 | Armitage witness proxy | **Complete** |
| 5 | Armitage OCI container (nix2gpu) | **Complete** |
| 6 | isospin TAP networking | **Complete** |
| 7 | **NativeLink CAS integration** | **Next** |
| 8 | **Graded monad executor** | Design |
| 9 | Nix binary cache facade | Planned |
| 10 | Lean proof discharge | Planned |
| 11 | Dhall rule schemas | Draft |
| 12 | R2 store backend | Planned |
| 13 | DICE (Buck2 fork) | Planned |
| 14 | stochastic_omega tactic | Research |
| 15 | nvidia.ko in Lean | Research |

## Component Index

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This document - overview |
| [armitage.md](armitage.md) | Nix compatibility layer, coeffects, graded monad execution |
| [dhall-bridge.md](dhall-bridge.md) | Dhall → Buck2 translation options |
| [ROADMAP.md](ROADMAP.md) | Timeline and milestones |

## Bucket Layout

All witnessed builds store artifacts in a unified content-addressed bucket:

```
r2.straylight.cx/
├── specs/              # Build computations (Lean/Dhall)
│   └── {hash}.lean
├── traces/             # Execution traces (CBOR)
│   └── {hash}.cbor  
├── cas/                # Content-addressed blobs (outputs + fetches)
│   └── {hash}
├── proofs/             # Compiled Lean discharge proofs
│   └── {hash}.olean
└── attestations/       # Signed attestations
    └── {hash}.json
```

The proxy is both **witness** (records fetches) and **substitutor** (serves
cached content). Same infrastructure, two modes.

## Infrastructure

| Service | Domain | Purpose |
|---------|--------|---------|
| DNS | ns1.straylight.cx | Dynamic DNS from attestation store |
| Resolver | resolve.straylight.cx | Name → CAS redirect for `nix run` |
| CAS | cas.straylight.cx | NativeLink gRPC endpoint |
| Storage | r2.straylight.cx | Cloudflare R2 (S3-compatible) |
| Git | git.straylight.cx | Attestation repository |

## Naming

```
ca://sha256:abc...                  # content-addressed (sovereign)
att://straylight.cx/llvm@18         # attested package
nix://nixpkgs#hello                 # legacy compat (via registry redirect)
```

## References

- RFC-007: Nix Formalization
- [Dhall Language](https://dhall-lang.org/)
- [Buck2](https://buck2.build/)
- [Firecracker](https://firecracker-microvm.github.io/)
- [Cloudflare R2](https://developers.cloudflare.com/r2/)
- [Lean4](https://lean-lang.org/)
- [libgit2](https://libgit2.org/)
- Petricek et al., "Coeffects: A calculus of context-dependent computation"
- Orchard et al., "Quantitative Type Theory"
