# Flake Modules Reference

Source: `nix/modules/flake/`

## Available Modules

| Module | Description |
|--------|-------------|
| `default` | All modules (recommended) |
| `default-with-demos` | All modules + prelude demos |
| `formatter` | Treefmt configuration |
| `lint` | Statix + deadnix linting |
| `docs` | mdBook documentation |
| `std` | Core overlays and aleph namespace |
| `devshell` | Development shell |
| `prelude` | Straylight prelude access |
| `prelude-demos` | Prelude demo packages |
| `nv-sdk` | NVIDIA SDK configuration |
| `container` | Container utilities |
| `nixpkgs` | Nixpkgs configuration |
| `nix-conf` | Nix configuration |
| `options-only` | Options schema (for docs generation) |

## Usage

```nix
{
  inputs.aleph.url = "github:straylight-software/aleph";

  outputs = inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      # ...
    };
}
```

## Options

### aleph.formatter

```nix
aleph.formatter = {
  enable = true;           # Enable formatter (default: true)
  indent-width = 2;        # Indent width (default: 2)
  line-length = 100;       # Max line length (default: 100)
  enable-check = true;     # Enable flake check (default: true)
};
```

### aleph.docs

```nix
aleph.docs = {
  enable = true;
  title = "Documentation";
  description = "";
  theme = "ono-sendai";    # "ono-sendai" (dark) or "maas" (light)
  src = ./docs;            # mdBook source directory
  options-src = ./docs;    # NDG options directory
  modules = [ ];           # NixOS modules for options extraction
};
```

### aleph.nixpkgs

```nix
aleph.nixpkgs = {
  nv = {
    enable = false;            # Enable NVIDIA support
    capabilities = ["8.9" "9.0"];  # GPU capabilities
    forward-compat = false;
  };
  allow-unfree = true;        # Allow unfree packages
};
```

### aleph.overlays

```nix
aleph.overlays = {
  enable = true;              # Enable aleph overlays (default: true)
  extra = [ ];                # Additional overlays
};
```

### aleph.devshell

```nix
aleph.devshell = {
  enable = false;
  nv.enable = false;          # Include NVIDIA tools
  extra-packages = pkgs: [ ]; # Additional packages
  extra-shell-hook = "";      # Additional shell hook
};
```

## Module Outputs

### default

Sets up:

- Formatter (treefmt with nixfmt-rfc-style, stylua, taplo, shfmt)
- Linter (statix + deadnix)
- Documentation (mdBook)
- Overlays (all aleph overlays)
- Devshell (development environment)
- Prelude (functional library access)
- Container (namespace utilities)
- NV-SDK (NVIDIA SDK configuration)

### nixpkgs

Configures nixpkgs with:

- CUDA support (if enabled)
- Allow unfree (configurable)
- Straylight overlays

### std

Applies overlays to pkgs:

- `aleph.prelude`
- `aleph.platform`
- `aleph.gpu`
- `aleph.stdenv`
- `aleph.cross`
- `aleph.container`
- `aleph.script`

### prelude

Exposes `config.aleph.prelude` containing:

- All prelude functions
- `stdenv` matrix
- `cross` targets
- `fetch` helpers
- `render` helpers
- `license` definitions
- Language toolchains (`python`, `ghc`)

### container

Provides container utilities:

- `mk-namespace-env`
- `mk-oci-rootfs`
- `mk-firecracker-image`
- `fhs-run`
- `gpu-run`
- `unshare-run`
