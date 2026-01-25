# Templates Reference

Source: `nix/templates/`

Available templates for `nix flake init`.

## default

Full aleph setup with formatter, linter, devshell, and prelude.

```bash
nix flake init -t github:straylight-software/aleph
```

Features:

- All flake modules imported
- Formatter enabled
- Devshell enabled
- Prelude access via `config.aleph.prelude`

## minimal

Just nixpkgs with aleph overlays. No formatter, no devshell.

```bash
nix flake init -t github:straylight-software/aleph#minimal
```

Features:

- Only `nixpkgs` module imported
- Minimal configuration
- Good for adding straylight to existing projects

## nv

NVIDIA development setup with CUDA SDK.

```bash
nix flake init -t github:straylight-software/aleph#nv
```

Features:

- NVIDIA SDK enabled
- CUDA packages available
- GPU stdenvs accessible

## dhall-configured

Configuration via Dhall instead of Nix.

```bash
nix flake init -t github:straylight-software/aleph#dhall-configured
```

## nickel-configured

Configuration via Nickel instead of Nix.

```bash
nix flake init -t github:straylight-software/aleph#nickel-configured
```

## Template Contents

### default/flake.nix

```nix
{
  description = "Project powered by aleph";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
  };

  outputs = inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];

      aleph = {
        formatter.enable = true;
        devshell.enable = true;
        nixpkgs.allow-unfree = true;
      };

      perSystem = { config, pkgs, ... }:
        let
          inherit (config.aleph) prelude;
        in {
          packages.default = pkgs.hello;
        };
    };
}
```

### minimal/flake.nix

```nix
{
  description = "Minimal project with aleph nixpkgs";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
  };

  outputs = inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.nixpkgs ];
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];

      aleph.nixpkgs.allow-unfree = true;

      perSystem = { pkgs, ... }: {
        packages.default = pkgs.hello;
      };
    };
}
```
