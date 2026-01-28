# RFC-012: render.nix - Typed Shell Scripts

## Status

Draft

## Abstract

A type system, LSP, formatter, and linter for bash scripts embedded in Nix. Enforces a safe subset of bash with no heredocs, no bare commands, no eval. Provides IDE features: hover types, go-to-definition, completions. Generates documentation from inferred types.

## Motivation

### The Problem

Bash scripts in Nix are write-only:

```nix
pkgs.writeShellScriptBin "deploy" ''
  PORT=''${PORT:-8080}
  cat << EOF > /tmp/config.json
  {
    "port": $PORT,
    "host": "''${HOST:-localhost}"
  }
  EOF
  curl -X POST http://localhost:$PORT/deploy -d @/tmp/config.json
''
```

What's wrong:
1. **Heredoc** - Three escaping systems (Nix, bash, JSON). Injection bugs. Untyped.
2. **Bare command** - `curl` could be `/usr/bin/curl` or nothing. Not reproducible.
3. **No types** - Is `PORT` a string or int? What's required? What's optional?
4. **No tooling** - No hover, no go-to-def, no completions, no docs.

### The Solution

Enforce a typed subset. Reject the bad patterns at commit time. Provide IDE support for the good patterns.

```nix
pkgs.writeShellScriptBin "deploy" ''
  # render.nix infers:
  #   PORT : Int = 8080
  #   HOST : String = "localhost"
  
  PORT="''${PORT:-8080}"
  HOST="''${HOST:-localhost}"
  
  ${pkgs.curl}/bin/curl -X POST \
    "http://$HOST:$PORT/deploy" \
    -d "$(${pkgs.dhall-json}/bin/dhall-to-json <<< '{ status = "starting" }')"
''
```

What's right:
1. **No heredoc** - Dhall for structured data, type-checked
2. **Store paths** - `${pkgs.curl}` pins the exact version
3. **Inferred types** - PORT is Int (from `8080`), HOST is String
4. **Full tooling** - Hover, completions, docs, all work

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         render.nix                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Nix Parser │───▶│ Bash Parser │───▶│    Type     │     │
│  │   (hnix)    │    │ (ShellCheck)│    │  Inference  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────┐       │
│  │                 Unified AST                      │       │
│  │  - Nix expressions with interpolation sites     │       │
│  │  - Bash AST with store path references          │       │
│  │  - Type annotations on all variables            │       │
│  └─────────────────────────────────────────────────┘       │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐       │
│  │   LSP     │      │ Formatter │      │  Linter   │       │
│  │  Server   │      │           │      │           │       │
│  └───────────┘      └───────────┘      └───────────┘       │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐       │
│  │  Hover    │      │  Pretty   │      │ Pre-commit│       │
│  │  Go-to-def│      │  Print +  │      │   Gate    │       │
│  │  Complete │      │  Types    │      │           │       │
│  └───────────┘      └───────────┘      └───────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Type System

#### Base Types

```
Type ::= TInt           -- integers: 8080, -1, 0
       | TString        -- strings: "hello", localhost
       | TBool          -- booleans: true, false
       | TPath          -- store paths: /nix/store/...
       | TArray Type    -- arrays: ("a" "b" "c")
       | TUnknown       -- not yet inferred
```

#### Inference Rules

```
─────────────────────────────────────────────────────
VAR="${VAR:-8080}"          ⊢  VAR : TInt

─────────────────────────────────────────────────────
VAR="${VAR:-true}"          ⊢  VAR : TBool

─────────────────────────────────────────────────────
VAR="${VAR:-hello}"         ⊢  VAR : TString

─────────────────────────────────────────────────────
VAR="${VAR:?}"              ⊢  VAR : TString, required

─────────────────────────────────────────────────────
VAR="${VAR:-$OTHER}"        ⊢  VAR : typeof(OTHER)

─────────────────────────────────────────────────────
${pkgs.foo}/bin/bar         ⊢  StorePath(/nix/store/...-foo)
```

#### Constraint Generation

Each pattern generates constraints. Unification solves them.

```haskell
data Constraint 
  = Type :~: Type           -- equality
  | Required Text           -- var must be provided
  | StorePath Text Path     -- interpolation resolves to path
  | BareCommand Text Span   -- error: unresolved command
  | Heredoc Span            -- error: heredoc detected
  | Eval Span               -- error: eval detected
  | Backtick Span           -- error: backtick detected
```

### Forbidden Constructs

These are errors, not warnings. No override flag.

#### 1. Heredocs

```bash
# FORBIDDEN
cat << EOF
{"port": $PORT}
EOF

# FORBIDDEN
cat << 'EOF'
literal text
EOF

# FORBIDDEN
cat <<< "here string with $VAR"
```

**Why**: Three escaping systems. Injection vulnerabilities. Untyped output.

**Instead**:
```bash
# Dhall for structured data
${dhall-json}/bin/dhall-to-json <<< '{ port = 8080 }'

# printf for simple strings
printf '{"port": %d}\n' "$PORT"

# Or generate at Nix level, not bash level
```

#### 2. Bare Commands

```bash
# FORBIDDEN
curl http://example.com
grep "pattern" file
jq '.foo'

# ALLOWED
${pkgs.curl}/bin/curl http://example.com
${pkgs.gnugrep}/bin/grep "pattern" file
${pkgs.jq}/bin/jq '.foo'
```

**Why**: Non-reproducible. Depends on `$PATH`. Version skew.

#### 3. eval

```bash
# FORBIDDEN
eval "$DYNAMIC_CODE"
eval "$(generate_commands)"
```

**Why**: Unanalyzable. Security risk. Type system can't help.

#### 4. Backticks

```bash
# FORBIDDEN
result=`some_command`

# ALLOWED
result=$(some_command)
```

**Why**: Deprecated syntax. Nesting issues. Confusing escaping.

#### 5. Unquoted Nix Interpolations

```nix
# FORBIDDEN (in Nix)
pkgs.writeShellScriptBin "foo" ''
  ${someVariable}  # might not be a store path
''

# ALLOWED
pkgs.writeShellScriptBin "foo" ''
  ${pkgs.curl}/bin/curl  # definitely a store path
''
```

**Why**: Interpolations must resolve to store paths, not arbitrary strings.

### LSP Server

#### Capabilities

| Feature | Nix Files | Bash in Nix |
|---------|-----------|-------------|
| Hover | Package info, type | Var type, default, required |
| Go to definition | Package def | Var definition |
| Find references | Usages | Var usages |
| Completion | Package names | Env vars, store paths |
| Diagnostics | Type errors | Heredocs, bare cmds |
| Code actions | Add store path | Fix bare command |

#### Hover Examples

Hovering over `${pkgs.curl}`:
```
curl 8.5.0

Outputs:
  bin: /nix/store/abc...-curl-8.5.0-bin
  dev: /nix/store/def...-curl-8.5.0-dev
  
From: nixpkgs#curl
```

Hovering over `$PORT`:
```
PORT : Int

Default: 8080
Required: no
Defined: line 5
Used: lines 8, 12, 15
```

Hovering over `curl` (bare command):
```
⚠ Bare command: curl

This command is not pinned to a store path.
Use: ${pkgs.curl}/bin/curl

[Quick Fix: Add store path]
```

#### Diagnostics

```
error[E001]: heredoc not allowed
 --> deploy.nix:5:3
  |
5 |   cat << EOF
  |   ^^^^^^^^^^ heredocs are forbidden
  |
  = help: use Dhall for structured output
  = help: use printf for simple strings

error[E002]: bare command
 --> deploy.nix:8:3
  |
8 |   curl http://example.com
  |   ^^^^ command not pinned to store path
  |
  = help: use ${pkgs.curl}/bin/curl

error[E003]: eval not allowed
 --> deploy.nix:12:3
   |
12 |   eval "$cmd"
   |   ^^^^ eval is forbidden
   |
   = help: refactor to avoid dynamic code execution
```

### Formatter

Takes bash (in Nix), infers types, outputs bash with type annotations.

#### Input

```nix
pkgs.writeShellScriptBin "deploy" ''
  PORT=''${PORT:-8080}
  HOST=''${HOST:-localhost}
  DB=''${DB_URL:?}
  ${pkgs.curl}/bin/curl "http://$HOST:$PORT"
''
```

#### Output

```nix
pkgs.writeShellScriptBin "deploy" ''
  # @env PORT : Int = 8080
  # @env HOST : String = "localhost"
  # @env DB_URL : String (required)
  # @uses curl : /nix/store/abc...-curl-8.5.0

  PORT="''${PORT:-8080}"
  HOST="''${HOST:-localhost}"
  DB="''${DB_URL:?}"
  
  ${pkgs.curl}/bin/curl "http://$HOST:$PORT"
''
```

The annotations are:
- Generated, not source of truth
- Re-derived on each format
- Machine-readable for doc generation

#### Formatting Rules

1. **Header block** with type signatures
2. **Blank line** after header
3. **Variable declarations** grouped at top
4. **Store path comment** showing resolved paths
5. **Consistent quoting** - always quote `"${VAR}"`
6. **Consistent spacing** - one space around operators

### Documentation Generator

Reads formatted scripts, emits markdown/HTML.

#### Output

```markdown
# deploy

Deployment script for the API service.

## Environment Variables

| Name | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `PORT` | Int | `8080` | No | Server port |
| `HOST` | String | `"localhost"` | No | Server host |
| `DB_URL` | String | - | **Yes** | Database connection URL |

## Dependencies

| Package | Version | Path |
|---------|---------|------|
| curl | 8.5.0 | `/nix/store/abc...-curl-8.5.0` |

## Source

\`\`\`bash
PORT="${PORT:-8080}"
HOST="${HOST:-localhost}"
DB="${DB_URL:?}"

${curl}/bin/curl "http://$HOST:$PORT"
\`\`\`
```

### Pre-commit Hook

Enforces everything. No bypass.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: render-check
        name: render.nix typecheck
        entry: render check
        language: system
        files: '\.nix$'
        pass_filenames: true
```

The hook:
1. Finds all `writeShellScript*` calls
2. Extracts and parses bash
3. Runs type inference
4. Checks for forbidden constructs
5. Fails commit if any errors

**No `--force` flag.** No `# render-ignore` comments. Fix the code or don't commit.

### Integration with Existing Tools

#### treefmt

```nix
# treefmt.nix
{
  programs.render = {
    enable = true;
    includes = [ "*.nix" ];
  };
}
```

#### nil/nixd (Nix LSP)

render.nix can run as a child process of nil/nixd, providing bash-specific features while the parent handles Nix.

Communication via LSP's `workspace/executeCommand` or a custom protocol.

#### ShellCheck

We use ShellCheck's parser but add:
- Type inference
- Store path tracking
- Stricter rules (no heredocs)
- Nix integration

ShellCheck warnings become errors where appropriate.

## Implementation

### Phase 1: Linter (Blocking)

Detect and reject:
- [ ] Heredocs (`<<`, `<<<`)
- [ ] Bare commands
- [ ] `eval`
- [ ] Backticks
- [ ] Unquoted Nix interpolations

This is the enforcement gate. Must work before anything else.

### Phase 2: Type Inference (Foundation)

- [ ] Parse Nix to find embedded bash
- [ ] Track interpolation sites and their Nix values
- [ ] Infer env var types from patterns
- [ ] Unification with error recovery
- [ ] Schema output (JSON)

### Phase 3: Formatter (Developer Experience)

- [ ] Pretty printer for bash-in-Nix
- [ ] Type annotation comments
- [ ] Store path comments
- [ ] Integration with treefmt

### Phase 4: LSP (Full IDE)

- [ ] Hover for types
- [ ] Go to definition
- [ ] Find references
- [ ] Completions
- [ ] Diagnostics
- [ ] Code actions

### Phase 5: Documentation

- [ ] Markdown generator
- [ ] HTML generator
- [ ] Integration with mdbook
- [ ] Automatic README sections

## File Structure

```
nix/render/
├── app/
│   ├── render.hs          # CLI entry point
│   └── lsp.hs             # LSP server entry point
├── lib/
│   ├── Render.hs          # Main module
│   ├── Render/
│   │   ├── Types.hs       # Core types
│   │   ├── Nix/
│   │   │   ├── Parse.hs   # Nix parser (hnix wrapper)
│   │   │   ├── Extract.hs # Find bash in Nix
│   │   │   └── Interp.hs  # Track interpolations
│   │   ├── Bash/
│   │   │   ├── Parse.hs   # Bash parser (ShellCheck)
│   │   │   ├── Facts.hs   # Extract facts
│   │   │   └── Patterns.hs# Pattern matchers
│   │   ├── Infer/
│   │   │   ├── Constraint.hs
│   │   │   └── Unify.hs
│   │   ├── Lint/
│   │   │   ├── Heredoc.hs
│   │   │   ├── BareCmd.hs
│   │   │   ├── Eval.hs
│   │   │   └── Policy.hs
│   │   ├── Format/
│   │   │   ├── Pretty.hs
│   │   │   └── Annotate.hs
│   │   ├── LSP/
│   │   │   ├── Server.hs
│   │   │   ├── Hover.hs
│   │   │   ├── Complete.hs
│   │   │   └── Diagnostic.hs
│   │   └── Doc/
│   │       ├── Markdown.hs
│   │       └── Html.hs
└── test/
    ├── Props.hs           # Property tests
    └── Golden/            # Golden tests
```

## CLI

```
render - typed shell scripts

USAGE:
    render <COMMAND>

COMMANDS:
    check       Typecheck and lint (exit 1 on errors)
    fmt         Format with type annotations
    infer       Output inferred schema (JSON)
    docs        Generate documentation
    lsp         Run LSP server

OPTIONS:
    --help      Show help
    --version   Show version

EXAMPLES:
    render check *.nix              # Lint all Nix files
    render fmt --write *.nix        # Format in place
    render infer deploy.nix         # Show schema
    render docs --out docs/ *.nix   # Generate docs
```

## Examples

### Before: Untyped Heredoc Hell

```nix
{ pkgs }:

pkgs.writeShellScriptBin "deploy" ''
  PORT=''${PORT:-8080}
  HOST=''${HOST:-localhost}
  
  cat << EOF > /tmp/config.json
  {
    "server": {
      "port": $PORT,
      "host": "$HOST"
    },
    "database": {
      "url": "''${DB_URL}"
    }
  }
  EOF
  
  curl -X POST http://admin:''${ADMIN_PASS}@$HOST:$PORT/deploy \
    -d @/tmp/config.json
''
```

Problems:
- Heredoc with mixed escaping
- Password in URL (visible in logs)
- Bare `curl` command
- Config file in /tmp (race condition)
- No type information

### After: Typed and Safe

```nix
{ pkgs }:

pkgs.writeShellScriptBin "deploy" ''
  # @env PORT : Int = 8080
  # @env HOST : String = "localhost"
  # @env DB_URL : String (required)
  # @env ADMIN_PASS : String (required)
  # @uses curl dhall-json

  set -euo pipefail
  
  PORT="''${PORT:-8080}"
  HOST="''${HOST:-localhost}"
  DB_URL="''${DB_URL:?}"
  ADMIN_PASS="''${ADMIN_PASS:?}"
  
  config=$(${pkgs.dhall-json}/bin/dhall-to-json << 'DHALL'
    { server = { port = env:PORT, host = env:HOST }
    , database = { url = env:DB_URL }
    }
  DHALL
  )
  
  ${pkgs.curl}/bin/curl \
    --netrc-file <(printf 'machine %s login admin password %s\n' "$HOST" "$ADMIN_PASS") \
    -X POST "http://$HOST:$PORT/deploy" \
    -H 'Content-Type: application/json' \
    -d "$config"
''
```

Improvements:
- Dhall for config (typed, no escaping issues)
- netrc for credentials (not in URL)
- Store path for curl
- Type annotations (generated)
- Config in variable (no temp file)

Wait, that still has a heredoc (`<< 'DHALL'`). Let me fix:

### After: Actually Correct

```nix
{ pkgs }:

let
  # Config schema in Nix/Dhall, not bash
  configDhall = pkgs.writeText "config.dhall" ''
    { server = { port = env:PORT, host = env:HOST }
    , database = { url = env:DB_URL }
    }
  '';
in
pkgs.writeShellScriptBin "deploy" ''
  # @env PORT : Int = 8080
  # @env HOST : String = "localhost"
  # @env DB_URL : String (required)
  # @env ADMIN_PASS : String (required)
  # @uses curl dhall-json

  set -euo pipefail
  
  PORT="''${PORT:-8080}"
  HOST="''${HOST:-localhost}"
  DB_URL="''${DB_URL:?}"
  ADMIN_PASS="''${ADMIN_PASS:?}"
  
  export PORT HOST DB_URL  # for Dhall env: references
  
  config=$(${pkgs.dhall-json}/bin/dhall-to-json --file ${configDhall})
  
  ${pkgs.curl}/bin/curl \
    -u "admin:$ADMIN_PASS" \
    -X POST "http://$HOST:$PORT/deploy" \
    -H 'Content-Type: application/json' \
    -d "$config"
''
```

Now:
- Config is a separate Dhall file (Nix handles interpolation)
- No heredocs in bash at all
- Credentials via `-u` (still visible in ps, but better than URL)
- All commands are store paths
- Type annotations generated from inference

## FAQ

### Why no heredocs at all?

Heredocs mix three languages (Nix, bash, target format). Each has its own escaping rules. The result is always wrong in subtle ways.

```nix
''
  cat << EOF
  {"path": "''${PATH//\\/\\\\}"}
  EOF
''
```

Is this correct? Who knows. Use Dhall.

### What about simple cases?

```bash
cat << EOF
Hello, $NAME
EOF
```

Still no. Use:

```bash
printf 'Hello, %s\n' "$NAME"
```

Or:

```bash
echo "Hello, $NAME"
```

Heredocs are never necessary. They're a convenience that costs clarity.

### What if I need multiline strings?

```nix
let
  message = pkgs.writeText "message.txt" ''
    Line 1
    Line 2
    Line 3
  '';
in
pkgs.writeShellScriptBin "foo" ''
  cat ${message}
''
```

Generate the content in Nix. Reference it in bash.

### What about <<< (here strings)?

Also forbidden. Use:

```bash
echo "string" | command
# or
command < <(echo "string")
# or
printf '%s' "string" | command
```

### Can I disable checks for legacy code?

No.

### Can I add ignore comments?

No.

### Can I use --force?

No.

Fix the code.

## References

- [ShellCheck](https://www.shellcheck.net/) - Bash parser and linter
- [hnix](https://github.com/haskell-nix/hnix) - Nix parser in Haskell
- [Dhall](https://dhall-lang.org/) - Typed configuration language
- [LSP Specification](https://microsoft.github.io/language-server-protocol/)
- [resholve](https://github.com/abathur/resholve) - Store path resolution for bash
