# Getting Started

## Prerequisites

- Nix 2.18+ with flakes enabled
- Git

## Enable Flakes

Add to `~/.config/nix/nix.conf` or `/etc/nix/nix.conf`:

```
experimental-features = nix-command flakes pipe-operators ca-derivations
```

n.b. you will have a much better time with `github:straylight-software/nix`, which supports all modern extensions out of the box including `WASM` derivations, and is being actively overhauled for modern C++23 patterns, performanace, and correctness. No political party affiliation required, no community conduct code witchhunts. Only rule is ship solid diffs.

## Templates

Initialize a new project:

```bash
# Full setup (formatter, linter, devshell, prelude)
nix flake init -t github:straylight-software/aleph

# Minimal (just nixpkgs with overlays)
nix flake init -t github:straylight-software/aleph#minimal

# NVIDIA development
nix flake init -t github:straylight-software/aleph#nv
```

## Your First Flake

Create `flake.nix`:

```nix
{
  inputs.aleph.url = "github:straylight-software/aleph";

  outputs = inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];

      perSystem = { config, pkgs, ... }: {
        packages.default = pkgs.hello;
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.ripgrep pkgs.fd ];
        };
      };
    };
}
```

Run:

```bash
nix build          # Build default package
nix develop        # Enter devshell
nix fmt            # Format all files
nix flake check    # Run all checks
```

## Using the Prelude

Access the prelude via `config.aleph.prelude`:

```nix
perSystem = { config, pkgs, ... }:
  let
    inherit (config.aleph) prelude;
    inherit (prelude) stdenv map filter fold;
  in {
    packages.my-tool = stdenv.default {
      pname = "my-tool";
      version = "0.1.0";
      src = ./.;
    };
  };
```

## Next Steps

- [Concepts](../concepts/README.md) - Understand the architecture
- [Guides](../guides/README.md) - Write packages, modules, scripts
- [Reference](../reference/README.md) - Look up specific functions
