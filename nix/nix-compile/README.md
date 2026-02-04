# // nix-compile //

Static analysis for the dynamic world of Nix and Bash.

*   **Type Inference**: Hindley-Milner typing for Nix expressions.
*   **Safe Bash**: Extract schema and validate usage of environment variables.
*   **Policy Enforcement**: Ban `with`, `rec`, and bare commands.

## // documentation //

See [docs/guide/nix-compile.md](../../docs/guide/nix-compile.md) for the full user guide.

## // quick start //

```bash
# Check a bash script
nix run .#nix-compile -- check ./deploy.sh

# Infer types for a Nix file
nix run .#nix-compile -- fmt ./default.nix
```
