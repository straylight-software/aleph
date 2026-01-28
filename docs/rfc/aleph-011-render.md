# RFC-011: render.nix

## Status

Draft

## Summary

Type inference for bash scripts. Extracts environment schemas, validates store path usage, generates typed Dhall configs.

## Motivation

Bash scripts are the glue of Nix. They configure services, orchestrate containers, wire up dependencies. But they're opaque:

- What env vars does this script need?
- Which are required vs optional?
- What are the types? The defaults?
- What commands does it call? Are they pinned to store paths?

Currently: read the script, grep for `${VAR:-`, hope the wiki is current.

render.nix makes scripts queryable. Parse once, get a typed schema. Use that schema to:
- Generate NixOS module options
- Generate Kubernetes manifests
- Generate documentation
- Validate at build time
- Emit typed config files

## Design

### Core Idea

Bash scripts already declare their schemas. We just parse them:

```bash
PORT="${PORT:-8080}"        # PORT : Int, default 8080
HOST="${HOST:-localhost}"   # HOST : String, default "localhost"  
DB_URL="${DB_URL:?}"        # DB_URL : String, required
DEBUG="${DEBUG:-false}"     # DEBUG : Bool, default false

${pkgs.curl}/bin/curl ...   # Depends on curl, pinned to store path
```

No annotations. No DSL. Just bash.

### Type Inference

Types are inferred from context:

| Pattern | Inferred Type |
|---------|---------------|
| `${VAR:-8080}` | Int (numeric literal) |
| `${VAR:-true}` / `${VAR:-false}` | Bool |
| `${VAR:-hello}` | String |
| `${VAR:?}` | String (required, no default) |
| `${VAR:-$OTHER}` | Same type as OTHER |

Unification resolves constraints. If `PORT` appears in `--port $PORT` and we know `--port` takes an int, `PORT` is an int.

### Schema Output

```nix
let
  schema = render.parse ./deploy.sh;
in {
  schema.env
  # => {
  #   PORT = { type = "Int"; default = 8080; required = false; };
  #   HOST = { type = "String"; default = "localhost"; required = false; };
  #   DB_URL = { type = "String"; required = true; };
  #   DEBUG = { type = "Bool"; default = false; required = false; };
  # }

  schema.storePaths
  # => [ "/nix/store/...-curl" ]

  schema.bareCommands
  # => [ "grep" ]  # Policy violation if requireStorePaths = true
}
```

### Dhall Config Generation

Instead of heredoc templating:

```bash
# BAD: Heredoc hell
cat > config.json << EOF
{
  "port": ${PORT},
  "host": "${HOST}",
  "debug": ${DEBUG}
}
EOF
```

Generate typed Dhall:

```nix
render.mkScript {
  name = "my-service";
  script = ''
    PORT="${PORT:-8080}"
    HOST="${HOST:-localhost}"
    DEBUG="${DEBUG:-false}"
    
    # emit-config generates Dhall, pipes to dhall-to-json
    ${pkgs.myapp}/bin/myapp --config <(emit-config json)
  '';
}
```

The `emit-config` function is auto-generated from the inferred schema:

```bash
emit-config() {
  local format="${1:-dhall}"
  
  # Generate Dhall source with actual values
  local dhall_src
  dhall_src=$(cat << 'DHALL'
{ port = env:PORT
, host = env:HOST  
, debug = env:DEBUG
}
DHALL
)
  
  case "$format" in
    dhall) echo "$dhall_src" ;;
    json)  echo "$dhall_src" | dhall-to-json ;;
    yaml)  echo "$dhall_src" | dhall-to-yaml ;;
    *)     echo "Unknown format: $format" >&2; return 1 ;;
  esac
}
```

Dhall handles:
- Type checking (port must be Natural, not String)
- Escaping (no injection bugs)
- Normalization (deterministic output)

### Store Path Enforcement

Scripts should use pinned store paths, not bare commands:

```nix
render.mkScript {
  name = "deploy";
  requireStorePaths = true;  # default
  script = ''
    ${pkgs.curl}/bin/curl ...  # OK: store path
    ${pkgs.jq}/bin/jq ...      # OK: store path
    grep ...                    # ERROR: bare command
  '';
}
```

Build fails with:

```
error: render.nix: bare command 'grep' at line 4
       
       Use a store path: ${pkgs.gnugrep}/bin/grep
       Or set: requireStorePaths = false;
```

### Policy Checks

Optional additional checks:

```nix
render.mkScript {
  name = "secure-script";
  policies = [ "no-eval" "no-backticks" "require-set-eu" ];
  script = ''
    set -eu
    ...
  '';
}
```

Policies:
- `no-eval`: No `eval` statements
- `no-backticks`: No `` `cmd` `` syntax (use `$(cmd)`)
- `require-set-e`: Must have `set -e`
- `require-set-u`: Must have `set -u`
- `require-set-eu`: Both

## API

### `render.parse`

Parse a script, return schema as Nix attrset.

```nix
render.parse :: Path -> {
  env : AttrsOf {
    type : String;      # "Int" | "String" | "Bool" | "Path"
    required : Bool;
    default : Opt Any;
    line : Int;
  };
  storePaths : ListOf Path;
  bareCommands : ListOf String;
  dynamicCommands : ListOf String;  # $VAR used as command
}
```

### `render.mkScript`

Build a script with analysis and emit-config injection.

```nix
render.mkScript :: {
  name : String;
  script : String;
  deps? : ListOf Derivation;      # Added to PATH
  requireStorePaths? : Bool;      # Default: true
  policies? : ListOf String;      # Policy checks
} -> Derivation
```

The built script has:
- `emit-config` function injected
- Dhall runtime available
- Schema in passthru: `drv.passthru.schema`

### `render.check`

Build-time check derivation. Fails if policy violations.

```nix
render.check :: Path -> Derivation
```

Use in flake checks:

```nix
checks.deploy-script = render.check ./deploy.sh;
```

## emit-config Specification

The injected `emit-config` function:

```bash
emit-config [format]

Formats:
  dhall   Raw Dhall source (default)
  json    Dhall -> JSON via dhall-to-json
  yaml    Dhall -> YAML via dhall-to-yaml
  bash    Bash variable exports

Examples:
  emit-config              # Dhall to stdout
  emit-config json         # JSON to stdout
  myapp --config <(emit-config json)
```

The Dhall source is generated at build time from the inferred schema:

```dhall
-- For env: PORT=8080, HOST=localhost, DEBUG=false
{ port = env:PORT
, host = env:HOST
, debug = env:DEBUG
}
```

Dhall's `env:VAR` syntax reads environment variables with type checking. If `PORT` isn't a valid Natural, Dhall fails with a type error.

## Implementation

### Phase 1: Core (Done)

- [x] ShellCheck-based parser
- [x] Fact extraction (DefaultIs, Required, AssignFrom, etc.)
- [x] HM type inference with unification
- [x] Schema building
- [x] CLI: parse, infer, check, emit
- [x] Property tests (50 tests)

### Phase 2: Integration

- [ ] `render.mkScript` builder
- [ ] emit-config injection with Dhall
- [ ] Store path enforcement
- [ ] Policy system

### Phase 3: Ecosystem

- [ ] `render.toNixOSOptions` - Generate module options from schema
- [ ] `render.toKubernetesEnv` - Generate K8s env specs
- [ ] `render.docs` - Generate documentation
- [ ] `render.diff` - Semantic diff between script versions

## Examples

### Basic Service Script

```nix
{ pkgs }:

pkgs.aleph.render.mkScript {
  name = "api-server";
  script = ''
    set -eu
    
    PORT="''${PORT:-8080}"
    HOST="''${HOST:-0.0.0.0}"
    DB_URL="''${DB_URL:?Database URL required}"
    LOG_LEVEL="''${LOG_LEVEL:-info}"
    
    exec ${pkgs.myapi}/bin/myapi \
      --config <(emit-config json) \
      "$@"
  '';
}
```

Results in:
- Schema with PORT (Int), HOST (String), DB_URL (required String), LOG_LEVEL (String)
- emit-config that generates `{ port: 8080, host: "0.0.0.0", ... }`
- Build fails if any bare commands

### NixOS Module from Script

```nix
{ config, lib, pkgs, ... }:

let
  script = pkgs.aleph.render.parse ./api-server.sh;
  
  # Auto-generate options from script's env schema
  envOptions = lib.mapAttrs (name: spec: lib.mkOption {
    type = if spec.type == "Int" then lib.types.int
           else if spec.type == "Bool" then lib.types.bool
           else lib.types.str;
    default = spec.default or null;
    description = "Auto-generated from ${name} in api-server.sh";
  }) script.env;

in {
  options.services.api-server = envOptions // {
    enable = lib.mkEnableOption "API server";
  };

  config = lib.mkIf config.services.api-server.enable {
    systemd.services.api-server = {
      script = "${script.build { name = "api-server"; }}/bin/api-server";
      environment = lib.mapAttrs (name: _: 
        toString config.services.api-server.${name}
      ) script.env;
    };
  };
}
```

## Alternatives Considered

### Annotation Comments

```bash
# @env PORT : Int = 8080
# @env DB_URL : String !required
PORT="${PORT:-8080}"
```

Rejected: Redundant. The bash already says this.

### Custom Config DSL

```bash
config.server.port=$PORT
config.server.host="$HOST"
```

Rejected: Not valid bash (ShellCheck errors), requires preprocessing.

### Associative Arrays

```bash
declare -A config
config["server.port"]=$PORT
```

Rejected: Valid bash, but complex to parse correctly from ShellCheck AST. The index handling varies.

### Keep it Simple

Just parse what bash already gives us. `${VAR:-default}` is the config DSL. Dhall is the output format.

## References

- [ShellCheck](https://www.shellcheck.net/) - Bash parser
- [Dhall](https://dhall-lang.org/) - Typed configuration language
- [resholve](https://github.com/abathur/resholve) - Prior art for store path resolution
