# // nix-compile // internals //

> *how the sausage is typed*

`nix-compile` is not a standard linter. It doesn't just regex your code. It parses it into an AST, extracts semantic facts, generates type constraints, and solves them using unification (Hindley-Milner).

## // architecture //

The tool is written in Haskell and consists of two main pipelines:

1.  **Bash Pipeline**: `NixCompile.Bash`
    *   **Parser**: Wraps `ShellCheck` to get a robust AST.
    *   **Fact Extractor**: Walks the AST to find assignments, commands, and `config.*` patterns.
    *   **Schema Builder**: Converts facts into a JSON schema (environment variables, config keys).

2.  **Nix Pipeline**: `NixCompile.Nix`
    *   **Parser**: Uses `hnix` to parse Nix expressions.
    *   **Infer**: Generates type constraints from the expression tree.
    *   **Unify**: Solves constraints (e.g. `t1 ~ Int`, `t1 ~ t2` -> `t2 ~ Int`).
    *   **Format**: Injects resolved types back into the source code as comments.

## // theory //

### The Type System

The type system is a subset of Nix, focused on what's verifyable at build time.

```haskell
data NixType
  = TInt               -- 42
  | TString            -- "hello"
  | TPath              -- ./foo
  | TBool              -- true
  | TList NixType      -- [ 1 2 3 ] (homogeneous)
  | TAttrs (Map Text NixType)  -- { x = 1; } (closed rows)
  | TFun NixType NixType       -- x: x + 1
  | TDerivation        -- pkgs.hello
  | TVar Int           -- Polymorphic variable (a, b...)
```

### Constraint Generation

We traverse the AST and emit constraints.

*   **Literal**: `42` -> `TInt`
*   **Application**: `f x`. If `f :: t1`, `x :: t2`, result `t3`. Constraint: `t1 ~ (t2 -> t3)`.
*   **List**: `[ x y ]`. If `x :: t1`, `y :: t2`. Constraint: `t1 ~ t2`. Result `[t1]`.

### Unification

We use a standard union-find algorithm to solve constraints. If we find a contradiction (e.g. `Int ~ String`), we report a type error.

## // limitations //

### Dynamic Scope (`with`)
The `with` construct introduces dynamic scoping, making static analysis impossible without evaluating the expression (which we don't do). `nix-compile` forces explicit scope.

**Bad:**
```nix
with pkgs; [ hello git ]
```

**Good:**
```nix
[ pkgs.hello pkgs.git ]
```

### Recursion (`rec`)
`rec` sets allow self-reference, which complicates type inference and can lead to infinite recursion. We prefer `let` bindings which separate definition from construction.

**Bad:**
```nix
rec {
  x = 1;
  y = x + 1;
}
```

**Good:**
```nix
let
  x = 1;
  y = x + 1;
in {
  inherit x y;
}
```
