# RFCs

Design decisions and rationale for aleph.

## RFC Index

| RFC | Title | Status |
|-----|-------|--------|
| [001](./aleph-001-standard-nix.md) | Standard Nix | Implemented |
| [002](./aleph-002-lint.md) | Linting | Implemented |
| [003](./aleph-003-prelude.md) | The Prelude | Implemented |
| [004](./aleph-004-typed-unix.md) | Aleph.Script | Implemented (Part I-II), In Progress (Part III: Zero-Bash) |
| [005](./aleph-005-profiles.md) | Nix Profiles | Implemented |
| [006](./aleph-006-safe-bash.md) | Safe Bash | **Superseded** by zero-bash architecture |
| [007](./aleph-007-formalization.md) | Nix Formalization | In Progress (Dhall + aleph-exec) |

## Core Principles

1. **Zero Bash** - No shell strings in derivations. Typed actions only.
1. **Dhall Validation** - Store paths validated against Nix store at eval time.
1. **Direct Execution** - `aleph-exec` runs actions, not bash interpreters.
1. **Typed Everything** - Haskell/PureScript → WASM → Dhall → validated derivation.

See also: [Conformance Report](./conformance-report.md)

## RFC Format

Each RFC includes:

1. **Summary** - One paragraph overview
1. **Motivation** - Why this is needed
1. **Design** - How it works
1. **Implementation** - Where to find the code
1. **Drawbacks** - Known limitations
1. **Alternatives** - Other approaches considered
