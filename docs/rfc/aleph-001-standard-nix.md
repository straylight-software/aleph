# ℵ-001: Straylight Standard Nix

| Field | Value |
|-------|-------|
| RFC | ℵ-001 |
| Title | Straylight Standard Nix |
| Author | Straylight |
| Status | Accepted |
| Created | 2025-01-05 |

## Abstract

This RFC establishes **Straylight Standard Nix**, a specification for writing Nix code within the
straylight-software organization. It codifies deliberate divergences from nixpkgs conventions where those
conventions do not serve our needs, and establishes normative requirements for code style,
module structure, and composition patterns.

## Motivation

Nix is the correct foundation for reproducible infrastructure. The model—content-addressed
storage, hermetic builds, declarative configuration—is sound. However, the stylistic
conventions of nixpkgs are not load-bearing. They are thirty years of accretion by thousands
of contributors with no central authority on style.

We are not nixpkgs. We are a vertically integrated organization building GPU inference
infrastructure. Our code does not need to merge upstream. Our conventions do not need to
match theirs. We have the luxury of internal consistency that a community project cannot
enforce.

This RFC establishes the authority to diverge and specifies where and why we do so.

## Specification

### 1. Naming Convention: lisp-case

All identifiers within Straylight Standard Nix SHALL use `lisp-case` (kebab-case):

```nix
# Conformant
straylight-api-server
config.aleph.services.inference-server.model-path
nix/modules/nixos/gpu-worker-common.nix

# Non-conformant
straylightApiServer
config.aleph.services.inferenceServer.modelPath
```

#### Exceptions

Standard Nix idioms that every practitioner knows:

```nix
{ pkgs, lib, config, ... }
final: prev:
inherit (lib) mkOption mkIf mkMerge;
```

Attributes interfacing with external APIs retain upstream names.

### 2. The Overlay as Universe Transformer

An overlay is a pure function from the world as it is to the world as it ought to be:

```nix
final: prev: { ... }
```

Overlays SHALL be:

1. **Minimal** — Each overlay does one thing
1. **Composed** — `lib.composeManyExtensions` combines them
1. **Centralized** — `aleph` owns the base overlays; projects extend

### 3. Central nixpkgs Configuration

All flakes consuming `aleph` SHALL use its nixpkgs configuration:

```nix
perSystem = { system, ... }: {
  _module.args.pkgs = inputs.aleph.nixpkgs.${system};
};
```

Per-flake nixpkgs configuration is FORBIDDEN.

### 4. Directory Structure

```
project/
├── flake.nix                    # Inputs and single import only
├── nix/
│   ├── main.nix                 # Top-level flake module
│   ├── flake-modules/           # Flake-parts modules
│   │   ├── something.nix        # Single-purpose module
│   │   └── complex-thing/       # Multi-file module
│   │       └── flake-module.nix # Aggregator
│   ├── nixos/                   # NixOS modules
│   │   ├── services/
│   │   └── hardware/
│   ├── overlays/                # Overlay compositions
│   │   ├── default.nix
│   │   └── packages/            # Callpackageables
│   ├── lib/                     # Utility libraries
│   ├── checks/                  # CI checks
│   └── configurations/
│       ├── nixos/               # Machine configs
│       └── home-manager/        # User configs
└── docs/
```

### 5. Module System Boundaries

| System | Location | Context |
|--------|----------|---------|
| flake-parts | `nix/flake-modules/` | Flake evaluation |
| NixOS | `nix/nixos/` | `lib.nixosSystem` |
| nix-darwin | `nix/darwin/` | `darwin.lib.darwinSystem` |
| home-manager | `nix/home/` | `home-manager.lib.homeManagerConfiguration` |

Flake modules follow these conventions:

- Single-purpose modules: `something.nix`
- Directory aggregators: `flake-module.nix`
- Default output: `default.nix` (only when module is named `default` in schema)

All non-flake-parts modules SHALL declare `_class`.

NixOS modules MAY also be flake modules if convenient, assigning into `flake.nixosModules`.

### 5.1 Callpackageables

**Packages** SHALL:

- Use `finalAttrs` where overlay participation is possible
- Have reasonable `passthru` (tests, updateScript, etc.)

**Libs** are callPackageable but:

- Take and return flat attrsets
- Are not required to have `passthru` or `finalAttrs`

**Checks** are callPackageable but:

- Do not need `finalAttrs` (not overlay-participating)

### 5.2 Overlays

Overlays are the principal way to publish callPackageables. Author discretion MAY decide
that `nix flake show` visibility or CI legibility merits separate publication.

### 5.3 Configuration

Unless at great need, configuration is typed as Dhall and consumed with nixpkgs primitives.

### 6. Forbidden Patterns

| Pattern | Reason |
|---------|--------|
| **Heredocs in inline bash** | High crimes and misdemeanors. See §6.1 |
| **Inline code >10 lines** | Untestable, unlintable. See §6.1 |
| `with lib;` | Obscures provenance, breaks tooling |
| `rec` in derivations | Breaks `overrideAttrs` |
| `if/then/else` in module config | Eager evaluation causes infinite recursion |
| Import from derivation | Forces builds during evaluation |
| `default.nix` in packages | Discards filename information |
| Per-flake nixpkgs config | Creates version mismatches |
| camelCase in aleph namespaces | Violates naming convention |
| Missing `_class` | Silent cross-module-system failures |
| Missing `meta` in packages | Breaks documentation and compliance |

#### 6.1 Code Generation and Inline Scripts

Inline scripts in Nix strings are the single largest source of bugs in Nix codebases.
They cannot be linted, cannot be type-checked, and encourage copy-paste proliferation
of untested code. The prelude provides typed alternatives.

**Hierarchy of preference (best to worst):**

1. **External files** — Put scripts in separate files, load with `builtins.readFile`
1. **AlephScript** — Typed inline scripting primitive (see §6.1.1)
1. **Prelude builders** — `prelude.write-shell-application`, `prelude.write-python-application`
1. **Never** — Heredocs (`cat <<EOF`), inline strings >10 lines

**Prelude builders** are escape hatches, not recommendations. They exist because
sometimes you need to generate a script. They are WARNING-level in aleph-lint.
If you find yourself using them frequently, you are doing something wrong.

```nix
# PREFERRED: External file
buildPhase = builtins.readFile ./build.sh;

# ACCEPTABLE: Prelude builder (WARNING)
script = prelude.write-shell-application {
  name = "my-script";
  runtime-inputs = [ pkgs.jq pkgs.curl ];
  text = ''
    curl -s "$1" | jq .
  '';
};

# FORBIDDEN: Heredoc in inline bash
buildPhase = ''
  cat > config.json << 'EOF'
  {"bad": "idea"}
  EOF
'';
```

##### 6.1.1 AlephScript

For cases requiring inline code generation with interpolation, use `prelude.aleph-script`:

```nix
configPhase = prelude.aleph-script {
  # Declarative file generation - no heredocs
  files."config.json" = builtins.toJSON {
    toolchain = {
      cc = "${llvm.clang}/bin/clang";
      cxx = "${llvm.clang}/bin/clang++";
    };
  };
  
  files.".buckconfig.local" = prelude.to-ini {
    cxx = {
      cc = "${llvm.clang}/bin/clang";
      ld = "${llvm.lld}/bin/ld.lld";
    };
  };
  
  # Shell commands (optional, runs after file generation)
  run = ''
    ln -sf ${prelude} prelude
  '';
};
```

AlephScript is the **only** sanctioned way to generate files inline. It:

- Separates file content from shell logic
- Uses structured data (`builtins.toJSON`, `prelude.to-ini`) instead of string templates
- Is mechanically verifiable by aleph-lint
- Provides a clear audit trail of what files are generated

The `files` attribute is a declarative specification. The `run` attribute is optional
and should be minimal—if your `run` exceeds 5 lines, extract it to a file.

### 7. Package Requirements

All packages SHALL:

1. Be callable by `callPackage`
1. Use `finalAttrs` pattern (not `rec`)
1. Provide `meta` with `description`, `license`, and `mainProgram` (if applicable)

```nix
{ lib, stdenv, fetchFromGitHub }:
stdenv.mkDerivation (finalAttrs: {
  pname = "my-tool";
  version = "1.0.0";
  meta = {
    description = "A tool";
    license = lib.licenses.mit;
    mainProgram = "my-tool";
  };
})
```

### 8. Documentation

Straylight Standard Nix documentation SHALL be generated using ndg.

Module options SHALL have descriptions that ndg can extract.

### 9. Mechanical Enforcement

Aleph Prelude requirements SHALL be mechanically enforced via `aleph-lint` as specified
in [ℵ-002](aleph-002-lint.md). CI pipelines SHALL fail on `aleph-lint` errors.

## Conformance

A flake is **Straylight Standard Nix conformant** if it:

1. Imports `aleph` and uses its centralized nixpkgs configuration
1. Follows the directory structure specified in §4
1. Adheres to the naming conventions specified in §1
1. Avoids all forbidden patterns specified in §6
1. Passes `nix flake check` and `nix fmt -- --check`
1. Passes `aleph-lint` with exit code 0

## Authority

This RFC is maintained by the infrastructure team at Straylight. Amendments require review
and approval. The specification is not a democracy—it is a dictatorship of clarity over
convention, consistency over compatibility, velocity over consensus.
