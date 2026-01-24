______________________________________________________________________

## name: straylight-script description: Write Haskell scripts using straylight-std's Aleph.Script infrastructure (Aleph.Script, Dhall configs, Nix integration)

# Aleph Script — Aleph.Script

straylight-std uses Haskell scripts instead of bash for complex operations. The `Aleph.Script` module provides a batteries-included prelude combining:

- **Shelly** for shell foundation (thread-safe, tracing, good errors)
- **Turtle-style** ergonomics (streaming, format, patterns)
- **Straylight-specific** helpers (GPU detection, containers, Nix integration)

## Directory Structure

```
nix/scripts/
├── Aleph/
│   ├── Script.hs                 # Main prelude (1193 lines)
│   ├── Nix.hs                    # WASM plugin interface
│   ├── Config/                   # Dhall type definitions
│   │   ├── Base.dhall            # StorePath, MemMiB, CpuCount, etc.
│   │   ├── Firecracker.dhall     # VM config schema
│   │   └── NvidiaSdk.dhall       # NVIDIA extraction config
│   └── Script/
│       ├── Config.hs             # FromDhall instances, newtypes
│       ├── Clap.hs               # Parser for clap-based CLI tools
│       ├── Getopt.hs             # Parser for GNU getopt_long tools
│       ├── Oci.hs                # OCI container operations
│       ├── Vm.hs                 # VM rootfs construction
│       ├── Vfio.hs               # GPU passthrough (VFIO/IOMMU)
│       ├── Vm/
│       │   └── Config.hs         # VM config types
│       ├── Nvidia/
│       │   └── Config.hs         # NVIDIA SDK config types
│       └── Tools/
│           ├── Tools.hs          # Re-exports all tool wrappers
│           ├── Rg.hs             # ripgrep (263 lines)
│           ├── Bwrap.hs          # bubblewrap (298 lines)
│           ├── Crane.hs          # OCI images (194 lines)
│           ├── Jq.hs             # JSON processor (235 lines)
│           └── ... (24 total)
├── nvidia-extract.hs             # NVIDIA SDK extraction
├── unshare-gpu.hs                # GPU container runner
├── isospin-run.hs                # Firecracker runner
├── vfio-bind.hs                  # GPU passthrough
└── ... (14 compiled scripts)
```

## Writing a New Script

### Step 1: Create the Script File

```haskell
-- nix/scripts/my-tool.hs
{-# LANGUAGE OverloadedStrings #-}

import Aleph.Script
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [arg1, arg2] -> script $ doThing (pack arg1) (pack arg2)
        _ -> script $ do
            echoErr "Usage: my-tool <arg1> <arg2>"
            exit 1

doThing :: Text -> Text -> Sh ()
doThing x y = do
    echoErr $ ":: Processing " <> x
    -- Shell commands
    output <- run "echo" [x, y]
    echo output
    -- Haskell logic
    when (x == "special") $ do
        echoErr ":: Special handling"
```

### Step 2: Register in script.nix

Edit `nix/overlays/script.nix`, add to the `compiled` attrset:

```nix
compiled = {
  # ... existing scripts ...

  my-tool = mkCompiledScript {
    name = "my-tool";
    deps = [
      final.jq        # Runtime dependencies
      final.crane
    ];
    # Optional: Dhall config (see below)
    # configExpr = ''{ field = "${somePath}" }'';
  };
};
```

### Step 3: Run It

```bash
# Compiled (production)
nix run .#straylight.script.compiled.my-tool -- arg1 arg2

# Interpreted (development, slower)
nix run .#straylight.script.shell
cd nix/scripts
runghc -i. my-tool.hs arg1 arg2
```

## Aleph.Script API Reference

### Running Scripts

```haskell
script      :: Sh a -> IO a   -- Silent, errors on failure
scriptV     :: Sh a -> IO a   -- Verbose (shows commands)
scriptDebug :: Sh a -> IO a   -- Very verbose
Sh                            -- Shell monad (from Shelly)
liftIO      :: IO a -> Sh a
```

### Running Commands

```haskell
run       :: FilePath -> [Text] -> Sh Text  -- Capture stdout
run_      :: FilePath -> [Text] -> Sh ()    -- Ignore stdout
bash      :: Text -> Sh Text                -- Run bash -c "..."
bash_     :: Text -> Sh ()
which     :: FilePath -> Sh (Maybe FilePath)
```

### Output Control

```haskell
echo      :: Text -> Sh ()       -- Print to stdout
echoErr   :: Text -> Sh ()       -- Print to stderr
die       :: Text -> Sh a        -- Print error and exit
exit      :: Int -> Sh ()
errExit   :: Bool -> Sh a -> Sh a -- Control error-on-failure
silently  :: Sh a -> Sh a         -- Suppress output
verbosely :: Sh a -> Sh a         -- Show commands
```

### FilePath Operations

```haskell
FilePath                          -- type FilePath = String
(</>), (<.>)                      -- Path construction
fromText   :: Text -> FilePath
toText     :: FilePath -> Maybe Text
toTextIgnore :: FilePath -> Text  -- Use this one
filename, dirname, basename, parent
```

### Filesystem

```haskell
ls, lsT, lsRecursive
cp, cpRecursive, mv, rm, rmRecursive
mkdir, mkdirP                     -- mkdir -p
pwd, cd, home
withTmpDir :: (FilePath -> Sh a) -> Sh a
symlink, readlink, canonicalize
test_f, test_d, test_e, test_s    -- Existence tests
```

### Text (from Data.Text)

```haskell
Text, pack, unpack
strip, lines, unlines, words, unwords
isPrefixOf, isSuffixOf, isInfixOf
replace, splitOn, breakOn
T.concat, T.take, T.drop, T.length, T.null
```

### Type-safe Formatting

```haskell
format :: Format a -> a           -- Returns Text
(%)    :: Format b -> Format a -> Format (a -> b)
s      -- Text
d      -- Int (decimal)
f      -- Double
w      -- Show a => a
fp     -- FilePath

-- Example:
format ("Found " % d % " files in " % fp) count dir
```

### Error Handling

```haskell
try, tryIO, catch, catchIO
bracket, finally, onException
ExitCode (..), lastExitCode
```

### JSON (from Aeson)

```haskell
Value (..), Object, Array
decode, encode, eitherDecode
(.:), (.:?), (.=), object
ToJSON (..), FromJSON (..)
```

### Concurrency

```haskell
async, wait, cancel, concurrently, race
sleep   :: Double -> Sh ()    -- Seconds
sleepMs :: Int -> Sh ()       -- Milliseconds
retry   :: Int -> Sh a -> Sh a
timeout :: Double -> Sh a -> Sh (Maybe a)
```

### GPU (Straylight-specific)

```haskell
data GpuArch = Volta | Turing | Ampere | Ada | Hopper | Blackwell
detectGpu    :: Sh (Maybe GpuArch)
withGpuBinds :: Sh [Text]     -- bwrap bind args for GPU
```

## Tool Wrappers

Import qualified to avoid name conflicts:

```haskell
import qualified Aleph.Script.Tools.Rg as Rg
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Aleph.Script.Tools.Crane as Crane
import qualified Aleph.Script.Tools.Jq as Jq
```

### Bwrap (Bubblewrap) — Builder Pattern

```haskell
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import Data.Function ((&))

let sandbox = Bwrap.defaults
        & Bwrap.roBind "/nix/store" "/nix/store"
        & Bwrap.dev "/dev"
        & Bwrap.proc "/proc"
        & Bwrap.tmpfs "/tmp"
        & Bwrap.setenv "HOME" "/root"
        & Bwrap.unshareAll
        & Bwrap.shareNet
        & Bwrap.dieWithParent

-- Execute (replaces process)
Bwrap.exec sandbox ["./myprogram", "--flag"]

-- Or capture output
output <- Bwrap.bwrap sandbox ["cat", "/etc/os-release"]
```

### Crane (OCI Images)

```haskell
import qualified Aleph.Script.Tools.Crane as Crane

-- Export container to directory
Crane.exportToDir Crane.defaults "alpine:latest" "/tmp/rootfs"

-- Get config JSON
configJson <- Crane.config "nginx:latest"

-- Get digest
digest <- Crane.digest "ubuntu:24.04"
```

### Rg (Ripgrep)

```haskell
import qualified Aleph.Script.Tools.Rg as Rg

-- Search with options
matches <- Rg.rg Rg.defaults { Rg.ignoreCase = True } "TODO" ["."]

-- Simple search
Rg.search "pattern" ["src/"]

-- Find files containing pattern
files <- Rg.searchFiles "TODO" ["."]
```

### Available Tool Wrappers

| Category | Tools |
|----------|-------|
| Clap (Rust) | rg, fd, bat, delta, dust, tokei, hyperfine, deadnix, statix, stylua, taplo, zoxide |
| GNU | ls, grep, sed, find, xargs, tar, gzip, wget, rsync |
| Hand-crafted | jq, crane, bwrap |

## Dhall Configuration

For scripts needing structured config from Nix:

### 1. Define Dhall Schema

```dhall
-- nix/scripts/Aleph/Config/MyTool.dhall
let Base = ./Base.dhall

let Config =
    { Type =
        { inputPath : Base.StorePath
        , outputPath : Text
        , verbose : Bool
        , threads : Base.CpuCount
        }
    , default =
        { verbose = False
        , threads = 4
        }
    }

in { Config }
```

### 2. Haskell FromDhall Types

```haskell
-- nix/scripts/Aleph/Script/MyTool/Config.hs
{-# LANGUAGE DeriveGeneric #-}
module Aleph.Script.MyTool.Config where

import Aleph.Script.Config (StorePath(..), loadConfigFile)
import Dhall (FromDhall, Generic)
import Data.Text (Text)
import Numeric.Natural (Natural)

data Config = Config
    { inputPath  :: StorePath
    , outputPath :: Text
    , verbose    :: Bool
    , threads    :: Natural
    } deriving (Show, Generic)

instance FromDhall Config

loadConfig :: FilePath -> IO Config
loadConfig = loadConfigFile
```

### 3. Use in Script

```haskell
-- nix/scripts/my-tool.hs
import Aleph.Script
import Aleph.Script.MyTool.Config
import System.Environment (lookupEnv)

main :: IO ()
main = do
    configPath <- lookupEnv "CONFIG_FILE"
    case configPath of
        Nothing -> putStrLn "Error: CONFIG_FILE not set" >> exitFailure
        Just path -> do
            cfg <- loadConfig path
            script $ runWithConfig cfg

runWithConfig :: Config -> Sh ()
runWithConfig Config{..} = do
    when verbose $ echoErr ":: Verbose mode"
    echoErr $ ":: Using " <> pack (show threads) <> " threads"
    -- inputPath is StorePath newtype, extract with unStorePath
    let input = unStorePath inputPath
    run_ "process" [input, outputPath]
```

### 4. Wire Up in Nix

```nix
my-tool = mkCompiledScript {
  name = "my-tool";
  deps = [ final.somePackage ];
  configExpr = ''
    { inputPath = "${someDerivation}/data"
    , outputPath = "/tmp/output"
    , verbose = true
    , threads = 8
    }
  '';
};
```

The wrapper sets `CONFIG_FILE` environment variable automatically.

## Script → Nix Derivation

### When to Use Haskell Scripts vs Pure Nix

**Use Haskell scripts** when:

- Complex logic (parsing, conditionals, error handling)
- Interactive/development workflows
- Ad-hoc exploration
- Runtime configuration needed

**Use pure Nix** when:

- Simple extraction/copying
- Reproducibility is critical (FOD)
- Build-time only (no runtime deps)
- Performance matters (avoid script compilation)

### Two-Stage Container Extraction Pattern

For extracting from OCI containers, use a two-stage approach:

**Stage 1: FOD to pull container (cached)**

```nix
containerRootfs = stdenvNoCC.mkDerivation {
  name = "container-rootfs";
  
  # Fixed-output derivation - network access allowed
  outputHashAlgo = "sha256";
  outputHashMode = "recursive";
  outputHash = "sha256-abc123...";
  
  nativeBuildInputs = [ crane gnutar gzip ];
  SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  
  buildCommand = ''
    mkdir -p $out
    crane export ${imageRef} - | tar -xf - -C $out
  '';
};
```

**Stage 2: Regular derivation to process**

```nix
extracted = stdenvNoCC.mkDerivation {
  name = "extracted";
  
  dontUnpack = true;
  
  installPhase = ''
    mkdir -p $out/lib
    # Copy from rootfs (use --no-preserve=mode for writable files!)
    cp -rL --no-preserve=mode ${containerRootfs}/path/to/libs/. $out/lib/
  '';
};
```

**Key gotchas:**

- Use `--no-preserve=mode` when copying from Nix store (avoids read-only permission errors)
- Use `-L` to dereference symlinks (containers have many)
- Use `dontUnpack = true` when referencing store path directly (not as `src`)

### Fixed-Output Derivation (FOD)

For downloads with known hash:

```nix
nvidia-sdk = stdenv.mkDerivation {
  name = "nvidia-sdk-${version}";
  
  outputHashAlgo = "sha256";
  outputHashMode = "recursive";
  outputHash = "sha256-abc123...";
  
  nativeBuildInputs = [ straylight.script.compiled.nvidia-extract ];
  
  buildCommand = ''
    nvidia-extract ${imageRef} $out
  '';
};
```

### Regular Derivation

When output depends on inputs:

```nix
processed = stdenv.mkDerivation {
  name = "processed";
  src = ./data;
  nativeBuildInputs = [ straylight.script.compiled.my-processor ];
  buildPhase = ''
    my-processor $src $out
  '';
};
```

## Argument Parsing Patterns

### Simple Positional Args

```haskell
main = do
    args <- getArgs
    case args of
        [image, output] -> script $ process (pack image) (pack output)
        _ -> script $ echoErr "Usage: tool <image> <output>" >> exit 1
```

### Flags + Positional

```haskell
data Args = Args
    { argVerbose :: Bool
    , argCpus :: Maybe Int
    , argImage :: Text
    , argCmd :: [Text]
    }

parseArgs :: [String] -> Args
parseArgs = go (Args False Nothing "" [])
  where
    go acc [] = acc
    go acc ("--verbose" : rest) = go acc{argVerbose = True} rest
    go acc ("--cpus" : n : rest) = go acc{argCpus = readMaybe n} rest
    go acc (x : rest)
        | "--" `isPrefixOf` x = go acc rest  -- skip unknown flags
        | T.null (argImage acc) = go acc{argImage = pack x} rest
        | otherwise = go acc{argCmd = argCmd acc ++ [pack x]} rest
```

## Domain Modules

### Aleph.Script.Oci — Container Operations

```haskell
import qualified Aleph.Script.Oci as Oci

-- Pull with caching
rootfs <- Oci.pullOrCache Oci.defaultConfig "alpine:latest"

-- Get container environment
env <- Oci.getContainerEnv "nvidia/cuda:13.0"

-- Build sandbox with GPU support
let sandbox = Oci.baseSandbox rootfs
        & Oci.withGpuSupport env gpuBinds
```

### Aleph.Script.Vm — VM Operations

```haskell
import qualified Aleph.Script.Vm as Vm

-- Build ext4 rootfs
Vm.buildExt4 "/tmp/rootfs-dir" "/tmp/rootfs.ext4"

-- Run Firecracker
Vm.runFirecracker Vm.defaultFirecrackerConfig
    { kernel = "/nix/store/.../vmlinux"
    , rootfs = "/tmp/rootfs.ext4"
    }
```

### Aleph.Script.Vfio — GPU Passthrough

```haskell
import qualified Aleph.Script.Vfio as Vfio

-- List NVIDIA GPUs
gpus <- Vfio.listNvidiaGpus

-- Bind to VFIO (returns all devices in IOMMU group)
devices <- Vfio.bindToVfio "0000:01:00.0"

-- Unbind
Vfio.unbindFromVfio "0000:01:00.0"
```

## Best Practices

1. **Always use `echoErr`** for status messages (keeps stdout clean for data)
1. **Use `errExit False`** when a command failing is acceptable
1. **Import tools qualified** to avoid name clashes
1. **Wrap external tools** in `Aleph/Script/Tools/` for reuse
1. **Parse args early**, fail fast with clear usage message
1. **Use `withTmpDir`** for intermediate files
1. **Use builder pattern** (`&`) for Bwrap sandboxes
1. **Patch ELF binaries** with `$ORIGIN` RPATHs for portability

## Generating Tool Wrappers

Auto-generate typed wrappers from CLI `--help`:

```bash
# Generate and print to stdout
nix run .#straylight.script.gen-wrapper -- rg

# Write to Tools/Fd.hs
nix run .#straylight.script.gen-wrapper -- fd --write

# Force GNU format
nix run .#straylight.script.gen-wrapper -- grep --gnu --write
```

## Existing Scripts Reference

| Script | Purpose | Runtime Deps |
|--------|---------|--------------|
| `nvidia-extract` | Extract NVIDIA SDK from NGC | crane, tar, patchelf, file |
| `unshare-gpu` | Run container with GPU | bwrap, crane, jq, pciutils |
| `unshare-run` | Run container (no GPU) | bwrap, crane, jq |
| `isospin-run` | Firecracker microVM | firecracker |
| `isospin-build` | Build VM rootfs | e2fsprogs, cpio, gzip |
| `cloud-hypervisor-run` | Cloud Hypervisor VM | cloud-hypervisor |
| `cloud-hypervisor-gpu` | CH with GPU passthrough | cloud-hypervisor, pciutils |
| `vfio-bind` | Bind GPU to VFIO | pciutils |
| `gpu-run` | Namespace with GPU | bwrap, pciutils |
