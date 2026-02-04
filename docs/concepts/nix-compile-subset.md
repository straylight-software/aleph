# // nix-compile // supported subset

The nix-compile type checker supports a **principled subset** of Nix that is amenable to high-effort static analysis.

## Supported Constructs

### 1. Pure Functions
- Lambda expressions: `x: x + 1`
- Pattern matching with closed records: `{ a, b }: ...`
- Pattern matching with open records: `{ a, b, ... }: ...`

### 2. Attribute Sets
- Construction: `{ a = 1; b = 2; }`
- Selection: `set.a` or `set."a-b"`
- Update (//): `set1 // set2` (right overrides left)
- Open records for width subtyping

### 3. Primitive Types
- Integers, Floats, Booleans, Strings, Paths
- Null
- Lists (homogeneous: `[Int]` not `[Int, String]`)

### 4. Control Flow
- Conditionals: `if cond then a else b`
- Let bindings: `let x = 1; in x`
- Recursive let: `let x = y; y = 1; in x`

### 5. Builtins
- All functions in `builtins.*` with known types
- `import` for static imports

## Explicitly Unsupported (Dynamic/Metaprogramming)

These constructs are **fundamentally incompatible** with static analysis:

### 1. NixOS Modules (`lib.evalModules`)
- Dynamic option merging
- Priority-based overrides (`mkForce`, `mkDefault`)
- Configuration construction at evaluation time

### 2. Dynamic Attribute Access
- `set.${variable}`
- `builtins.getAttr name set`

### 3. Dynamic Code Execution
- `builtins.exec`
- `import` of computed paths
- String interpolation in import paths: `import "./${file}.nix"`

### 4. Recursive Scoping Ambiguity
- `with expr;` (obscures variable sources)
- `rec { }` with forward references

### 5. Derivation Meta-Programming
- `mkDerivation` attribute merging
- `overrideAttrs`
- Complex fixed-point loops

## Type Checking Strategy

Files using unsupported constructs will receive:
- **Parse error**: If syntax is invalid
- **Type error**: If supported subset has type mismatch
- **Skip**: If unsupported constructs detected (with warning)

## Goal

This subset captures **90% of configuration logic** while remaining **100% analyzable**. Dynamic/metaprogramming constructs should be isolated to specific module files that are excluded from type checking.
