# // aleph // nix-compile //

> *static analysis for the dynamic world*

`nix-compile` is a toolchain for enforcing correctness in Nix and Bash codebases. It implements [RFC-006 (Safe Bash)](../rfc/aleph-006-safe-bash.md) and [RFC-007 (Nix Formalization)](../rfc/aleph-007-formalization.md).

## // features //

*   **Bash Static Analysis**: Infers variable types, detects bare commands, enforces store path usage, and extracts configuration schemas.
*   **Nix Type Inference**: Hindley-Milner type inference for Nix expressions. Detects type errors at build time.
*   **Policy Enforcement**: Bans dangerous constructs like `eval`, `with`, `rec`, and dynamic attribute access.
*   **Auto-Formatting**: Injects inferred type signatures into Nix source code as comments.

## // usage //

The tool is available in the `aleph` flake:

```bash
nix run .#nix-compile -- <command> [args]
```

### 1. Format & Type Nix Files

Automatically infer types and annotate your Nix code.

```bash
nix run .#nix-compile -- fmt ./path/to/file.nix
```

**Before:**
```nix
{ pkgs }:
let
  port = 8080;
  host = "localhost";
in { inherit port host; }
```

**After:**
```nix
{ pkgs }:
let
  # :: Int
  port = 8080;
  # :: String
  host = "localhost";
in { inherit port host; }
```

### 2. Check Bash Scripts

Ensure your shell scripts are safe, hermetic, and typed.

```bash
nix run .#nix-compile -- check ./deploy.sh
```

Checks performed:
*   **Forbidden Constructs**: No `eval`, `heredocs`, or `backticks`.
*   **Hermeticity**: All commands must be absolute store paths (e.g. `${pkgs.curl}/bin/curl`) or whitelisted utilities.
*   **Type Safety**: Variable usage is consistent (e.g. not using a string as an array).

### 3. Infer Script Schema

Extract the interface of a shell script as a JSON schema.

```bash
nix run .#nix-compile -- infer ./deploy.sh
```

Output:
```json
{
  "env": {
    "PORT": { "type": "TInt", "required": false, "default": "8080" },
    "HOST": { "type": "TString", "required": false, "default": "localhost" }
  },
  "config": {
    "server.port": { "type": "TInt" }
  }
}
```

### 4. Recursive Type Check

Validate an entire directory of Nix files.

```bash
nix run .#nix-compile -- typecheck ./nix
```

This reports:
*   **[OK]**: Type checked successfully.
*   **[XX]**: Type error found.
*   **[SKIP]**: File uses unsupported dynamic features (e.g. `with`).

## // best practices //

### Bash

*   **Use Store Paths**: Never rely on `PATH`. Use `${pkgs.foo}/bin/foo`.
*   **Structured Config**: Use `config.path.key="$VAR"` assignments to export schema.
*   **Typed Variables**: Use default values to hint types (`PORT="${PORT:-8080}"` implies Integer).

### Nix

*   **No `with`**: Always use explicit access (e.g. `pkgs.lib` instead of `with pkgs; lib`).
*   **No `rec`**: Use `let` bindings for mutual recursion.
*   **Static Keys**: Avoid `${dynamic}` attribute names.

## // examples //

See `examples/nix-compile/` for illustrative examples.

*   `deploy.sh`: A compliant bash script with typed environment variables.
*   `typed.nix`: A clean Nix file ready for type annotation.
