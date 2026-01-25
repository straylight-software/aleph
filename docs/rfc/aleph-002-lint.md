# ℵ-002: aleph-lint

| Field | Value |
|-------|-------|
| RFC | ℵ-002 |
| Title | aleph-lint — Machine-Checkable Aleph Prelude |
| Author | Straylight |
| Status | Draft |
| Created | 2025-01-05 |
| Requires | ℵ-001 |

## Abstract

This RFC specifies `aleph-lint`, a static analysis tool that mechanically enforces
the Aleph Prelude conventions. The tool uses ast-grep for parsing and AST analysis,
enabling most ℵ-001 requirements to be checked without evaluation.

## Motivation

A specification without enforcement is a suggestion. ℵ-001 establishes normative
requirements, but compliance currently depends on code review.

The linter is doctrine, not guardrails. Each rule is a forcing function that drives
code through the typed prelude boundary. Error messages are designed for both agents
and humans, optimized to prevent vendor lock-in.

## Specification

### Error Levels

| Level | Meaning |
|-------|---------|
| ERROR | Non-conformant, CI must fail |
| WARNING | Review required |
| INFO | Suggestion |

### AST Rules (No Evaluation Required)

#### ALEPH-E001: with-statement

**Level:** ERROR

```nix
# ERROR
with lib;
{ options.foo = mkOption { }; }

# OK (list context exception)
environment.systemPackages = with pkgs; [ vim git ];
```

#### ALEPH-E002: rec-in-derivation

**Level:** ERROR

```nix
# ERROR
stdenv.mkDerivation rec { version = "1.0"; }

# OK
stdenv.mkDerivation (finalAttrs: { version = "1.0"; })
```

#### ALEPH-E003: non-lisp-case

**Level:** ERROR

Identifiers in `aleph.*` namespaces using camelCase or snake_case.

#### ALEPH-E004: missing-class

**Level:** ERROR

Files in `nix/modules/(nixos|darwin|home)/` without `_class` attribute.

#### ALEPH-E005: default-nix-in-packages

**Level:** ERROR

```
# ERROR
nix/packages/my-tool/default.nix

# OK
nix/packages/my-tool.nix
```

#### ALEPH-E006: heredoc-in-inline-bash

**Level:** ERROR

Heredocs inside Nix inline strings are forbidden.

#### ALEPH-E007: substitute-all

**Level:** ERROR

`substituteAll`, `replaceVars`, and `substitute` are forbidden.
All text generation must use Dhall templates.

#### ALEPH-E010: raw-mkderivation

**Level:** ERROR

Direct `mkDerivation` calls bypass the typed prelude boundary.
Use `aleph.stdenv.*` instead.

#### ALEPH-E011: raw-runcommand

**Level:** ERROR

Direct `runCommand` calls bypass the typed prelude boundary.
Use `aleph.run-command` instead.

#### ALEPH-E012: raw-writeshellapplication

**Level:** ERROR

Direct `writeShellApplication` calls bypass the typed prelude boundary.
Use `aleph.write-shell-application` instead.

#### ALEPH-E013: translate-attrs-outside-prelude

**Level:** ERROR

`translate-attrs` is the prelude's internal translation layer.
Using it directly means you're bypassing the typed interface.

### Warnings

#### ALEPH-W001: rec-anywhere

**Level:** WARNING

Flag all `rec` attrsets for review.

#### ALEPH-W003: long-inline-string

**Level:** WARNING

Multi-line strings exceeding 10 lines.

#### ALEPH-W004: or-null-fallback

**Level:** WARNING

Using `x or null` hides errors instead of failing fast.

#### ALEPH-W005: missing-description

**Level:** WARNING

`mkOption` calls without `description` attribute.

#### ALEPH-W006: prefer-write-shell-application

**Level:** ERROR

`writeShellScript` and `writeShellScriptBin` are deprecated.
Use `writeShellApplication` (which runs shellcheck).

#### ALEPH-W007: missing-meta

**Level:** WARNING

Derivation builder calls without `meta` attribute.

## Usage

```bash
# Check current directory
aleph-lint .

# Check specific files
aleph-lint nix/packages/*.nix

# JSON output for CI
aleph-lint --json .
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Conformant |
| 1 | Errors found |
| 2 | Warnings only (with --strict) |

## Implementation

aleph-lint uses ast-grep for parsing:

```yaml
id: with-lib
language: nix
severity: error
rule:
  kind: with_expression
  has:
    field: environment
    kind: variable_expression
    has:
      kind: identifier
      regex: ^lib$
message: "ALEPH-E001: `with lib;` statement"
```

## CI Integration

```yaml
# .github/workflows/lint.yml
- name: aleph-lint
  run: |
    nix run github:straylight-software/aleph#aleph-lint -- .
```

## Future Work

- IDE integration (LSP)
- Auto-fix for more patterns
- Custom rule definitions
