# ℵ-002: wsn-lint

| Field | Value |
|-------|-------|
| RFC | ℵ-002 |
| Title | wsn-lint — Machine-Checkable Straylight Standard Nix |
| Author | Straylight |
| Status | Draft |
| Created | 2025-01-05 |
| Requires | ℵ-001 |

## Abstract

This RFC specifies `wsn-lint`, a static analysis tool that mechanically enforces
Straylight Standard Nix. The tool uses hnix for parsing and AST analysis, enabling most
ℵ-001 requirements to be checked without evaluation.

## Motivation

A specification without enforcement is a suggestion. ℵ-001 establishes normative
requirements, but compliance currently depends on code review.

## Specification

### Error Levels

| Level | Meaning |
|-------|---------|
| ERROR | Non-conformant, CI must fail |
| WARNING | Review required |
| INFO | Suggestion |

### AST Rules (No Evaluation Required)

#### WSN-E001: with-statement

**Level:** ERROR

```nix
# ERROR
with lib;
{ options.foo = mkOption { }; }

# OK (list context exception)
environment.systemPackages = with pkgs; [ vim git ];
```

#### WSN-E002: rec-in-derivation

**Level:** ERROR

```nix
# ERROR
stdenv.mkDerivation rec { version = "1.0"; }

# OK
stdenv.mkDerivation (finalAttrs: { version = "1.0"; })
```

#### WSN-E003: non-lisp-case

**Level:** ERROR

Identifiers in `straylight.*` namespaces using camelCase or snake_case.

#### WSN-E004: missing-class

**Level:** ERROR

Files in `nix/modules/(nixos|darwin|home)/` without `_class` attribute.

#### WSN-E005: default-nix-in-packages

**Level:** ERROR

```
# ERROR
nix/packages/my-tool/default.nix

# OK
nix/packages/my-tool.nix
```

### Warnings

#### WSN-W001: rec-anywhere

**Level:** WARNING

Flag all `rec` attrsets for review.

#### WSN-W002: if-in-module-config

**Level:** WARNING

`if/then/else` inside module `config` blocks.

#### WSN-W003: long-inline-string

**Level:** WARNING

Multi-line strings exceeding 10 lines.

#### WSN-W004: missing-meta

**Level:** WARNING

Derivation builder calls without `meta` attribute.

#### WSN-W005: missing-description

**Level:** WARNING

`mkOption` calls without `description` attribute.

### Import Graph Rules (Requires Resolution)

#### WSN-E010: external-nixpkgs-config

**Level:** ERROR

```nix
# ERROR
pkgs = import inputs.nixpkgs { config.cudaSupport = true; };

# OK
pkgs = inputs.aleph.nixpkgs.${system};
```

#### WSN-E011: nixpkgs-config-in-nixos

**Level:** ERROR

Assignment to `nixpkgs.config.*` in NixOS configuration files.

## Usage

```bash
# Check current directory
wsn-lint check .

# Check flake
wsn-lint check --flake .

# Auto-fix where possible
wsn-lint fix .

# JSON output for CI
wsn-lint check --format json .
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Conformant |
| 1 | Errors found |
| 2 | Warnings only (with --strict) |

## Implementation

wsn-lint uses hnix for parsing:

```haskell
import Nix.Parser (parseNixFileLoc)
import Nix.Expr.Types

checkWithStatement :: NExprLoc -> [Diagnostic]
checkWithStatement expr = case expr of
  NWith _ _ body -> [Error "WSN-E001" (spanOf expr) "with statement"]
  _ -> []
```

## CI Integration

```yaml
# .github/workflows/lint.yml
- name: wsn-lint
  run: |
    nix run github:straylight-software/aleph -- check --flake .
```

## Future Work

- IDE integration (LSP)
- Auto-fix for more patterns
- Custom rule definitions
