# Aleph.Script

Haskell scripts instead of bash. `Aleph.Script` provides a type-safe shell scripting environment.

## Why Not Bash?

Bash scripts fail silently, have arcane quoting rules, and can't be reasoned about statically. Haskell scripts:

- Type-check before running
- Have proper error handling (ExceptT/Either)
- Parse arguments with real types
- Handle paths safely (no word splitting)
- Compile to ~2ms startup (same as bash)

## Aleph.Script

Source: `nix/scripts/Aleph/Script.hs`

Combines:

- **Shelly**: Thread-safe, good tracing, proper errors
- **Turtle**: Streaming, format strings, patterns

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script

main :: IO ()
main = script $ do
    echo "Hello from Straylight!"
    files <- ls "."
    for_ files $ \f ->
        when (hasExtension "nix" f) $
            echo $ format ("Found: "%fp) f
```

## Key Functions

### Running Scripts

```haskell
script :: Sh a -> IO a           -- Run with defaults
scriptV :: Sh a -> IO a          -- Verbose (shows commands)
scriptDebug :: Sh a -> IO a      -- Very verbose (shows output)
```

### File Operations

```haskell
ls :: FilePath -> Sh [FilePath]
cd :: FilePath -> Sh ()
pwd :: Sh FilePath
mkdir :: FilePath -> Sh ()
mkdirP :: FilePath -> Sh ()      -- mkdir -p
rm :: FilePath -> Sh ()
rmRf :: FilePath -> Sh ()        -- rm -rf
cp :: FilePath -> FilePath -> Sh ()
mv :: FilePath -> FilePath -> Sh ()
test_f :: FilePath -> Sh Bool    -- file exists?
test_d :: FilePath -> Sh Bool    -- directory exists?
```

### Running Commands

```haskell
run :: FilePath -> [Text] -> Sh Text     -- Run, capture stdout
run_ :: FilePath -> [Text] -> Sh ()      -- Run, ignore output
bash :: Text -> Sh Text                  -- Run bash command
cmd :: FilePath -> [Text] -> Sh ()       -- Run with streaming
```

### Text Operations

```haskell
echo :: Text -> Sh ()
echoErr :: Text -> Sh ()
lines :: Text -> [Text]
unlines :: [Text] -> Text
strip :: Text -> Text
```

### Path Operations

```haskell
(</>) :: FilePath -> FilePath -> FilePath
(<.>) :: FilePath -> Text -> FilePath
filename :: FilePath -> FilePath
dirname :: FilePath -> FilePath
extension :: FilePath -> Maybe Text
hasExtension :: Text -> FilePath -> Bool
```

### Formatting (from Turtle)

```haskell
format :: Format Text a -> a
(%) :: Format b a -> Format c b -> Format c a

s :: Format r (Text -> r)        -- Text
d :: Format r (Int -> r)         -- Decimal
fp :: Format r (FilePath -> r)   -- FilePath
w :: Show a => Format r (a -> r) -- Show
```

Example:

```haskell
format ("Found "%d%" files in "%fp) 42 "/tmp"
-- => "Found 42 files in /tmp"
```

## Compiled Scripts

Source: `nix/scripts/*.hs`

32 Haskell scripts for system operations:

### Container Operations

- `unshare-run.hs` - Run OCI images in bwrap/unshare namespaces
- `unshare-gpu.hs` - Run with GPU device access
- `crane-inspect.hs` - Inspect OCI image metadata
- `crane-pull.hs` - Pull OCI images

### Namespace Runners

- `fhs-run.hs` - Run with FHS layout
- `gpu-run.hs` - Run with GPU access

### VM Operations

- `isospin-run.hs` - Run Firecracker VMs
- `isospin-build.hs` - Build Firecracker disk images
- `cloud-hypervisor-run.hs` - Run Cloud Hypervisor VMs
- `cloud-hypervisor-gpu.hs` - Run with GPU passthrough

### GPU Passthrough

- `vfio-bind.hs` - Bind PCI devices to vfio-pci
- `vfio-unbind.hs` - Unbind from vfio-pci
- `vfio-list.hs` - List VFIO-capable devices

### Development Tools

- `check.hs` - Validation script
- `gen-wrapper.hs` - Generate typed CLI wrappers
- `nix-dev.hs` - Development Nix wrapper
- `nix-ci.hs` - CI Nix wrapper

## Using in Nix

### Via Overlay

```nix
pkgs.aleph.script.ghc           # GHC with Aleph.Script
pkgs.aleph.script.compiled.unshare-run  # Pre-compiled script
```

### Via Prelude

```nix
perSystem = { config, ... }:
  let
    inherit (config.aleph.prelude) ghc;
  in {
    packages.my-script = ghc.turtle-script {
      name = "my-script";
      src = ./my-script.hs;
      deps = [ pkgs.crane pkgs.bubblewrap ];
      hs-deps = p: [ p.aeson p.optparse-applicative ];
    };
  };
```

## Development Shell

```bash
nix develop .#aleph-script
runghc -i. check.hs        # Quick validation
runghc -i. Props.hs        # Property tests
runghc -i. gen-wrapper.hs rg  # Generate wrapper for ripgrep
```
