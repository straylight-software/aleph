# Contributing to Aleph-Naught

## The Right Way

Three rules:

1. **No bash logic** — If it has a branch, it's not bash
1. **Types over strings** — Use the Haskell DSL, not string interpolation
1. **Static over dynamic** — pkg-config ABI, not cmake soup

## Where Does My File Go?

### Decision Tree

```
Is it a C++ library?
├─ Yes → nix/overlays/libmodern/
└─ No
   Is it a CLI tool wrapper?
   ├─ Yes → nix/scripts/Weyl/Script/Tools/
   └─ No
      Is it a compiled script (executable)?
      ├─ Yes → nix/scripts/ (top level)
      └─ No
         Is it a domain module (OCI, VFIO, VM)?
         ├─ Yes → nix/scripts/Weyl/Script/
         └─ No
            Is it a flake-parts module?
            ├─ Yes → nix/modules/flake/
            └─ No
               Is it a NixOS module?
               ├─ Yes → nix/modules/nixos/
               └─ No
                  Is it a pure function?
                  ├─ Yes → nix/prelude/functions/
                  └─ No → nix/overlays/packages/
```

### Quick Reference

| Type | Location | Example |
|------|----------|---------|
| Static C++ library | `nix/overlays/libmodern/` | `fmt.nix`, `abseil-cpp/` |
| CLI tool wrapper | `nix/scripts/Weyl/Script/Tools/` | `Rg.hs`, `Tar.hs` |
| Compiled script | `src/tools/scripts/` | `unshare-run.hs`, `vfio-bind.hs` |
| Domain module | `nix/scripts/Weyl/Script/` | `Oci.hs`, `Vfio.hs`, `Vm.hs` |
| Flake module | `nix/modules/flake/` | `default.nix`, `nv-sdk.nix` |
| NixOS module | `nix/modules/nixos/` | `nv-driver.nix` |
| Pure Nix function | `nix/prelude/functions/` | `fold.nix`, `merge.nix` |
| Package overlay | `nix/overlays/packages/` | `llvm-git.nix` |
| WASM package def | `nix/scripts/Weyl/Nix/Wasm/Packages/` | `Fmt.hs`, `AbseilCpp.hs` |

## Writing Code the Right Way

### Adding a libmodern Package

```nix
# nix/overlays/libmodern/mylib.nix
{ mk-static-cpp, fetchFromGitHub }:

mk-static-cpp {
  pname = "mylib";
  version = "1.0.0";

  src = fetchFromGitHub {
    owner = "example";
    repo = "mylib";
    rev = "v1.0.0";
    hash = "sha256-...";
  };

  # CMake flags (typed in the Haskell DSL, but plain strings here)
  cmakeFlags = [
    "-DBUILD_TESTING=OFF"
    "-DMYLIB_OPTION=ON"
  ];
}
```

Then add to `nix/overlays/libmodern/default.nix`:

```nix
libmodern = {
  # existing...
  mylib = callPackage ./mylib.nix { };
};
```

### Adding a CLI Tool Wrapper

1. Generate from `--help`:

```bash
nix run .#gen-wrapper -- mytool --write
```

2. Or write by hand in `nix/scripts/Weyl/Script/Tools/Mytool.hs`:

```haskell
module Weyl.Script.Tools.Mytool
  ( run, run_
  , verbose_, quiet_, output_
  ) where

import Weyl.Script (Text, FilePath, Sh)
import qualified Weyl.Script as S

-- | Run mytool with arguments
run :: [Text] -> [FilePath] -> Sh Text
run opts paths = S.run "mytool" (opts <> map S.toTextIgnore paths)

run_ :: [Text] -> [FilePath] -> Sh ()
run_ opts paths = S.run_ "mytool" (opts <> map S.toTextIgnore paths)

-- Typed option constructors
verbose_ :: Text
verbose_ = "--verbose"

quiet_ :: Text
quiet_ = "--quiet"

output_ :: FilePath -> Text
output_ p = "--output=" <> S.toTextIgnore p
```

### Adding a Compiled Script

Create `nix/scripts/my-script.hs`:

```haskell
#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

import Weyl.Script
import qualified Weyl.Script.Tools.Rg as Rg

main :: IO ()
main = script $ do
  args <- liftIO getArgs
  case args of
    [pattern] -> do
      results <- Rg.run [Rg.glob_ "*.hs"] ["."]
      echo results
    _ -> do
      echoErr "Usage: my-script <pattern>"
      exit 1
```

### Safe Bash (The Only Permitted Bash)

If you absolutely must write bash, it can only be:

```nix
writeShellApplication {
  name = "my-wrapper";
  runtimeInputs = [ myHaskellScript ];
  text = ''
    exec my-haskell-script "$@"
  '';
}
```

That's it. No variables. No conditionals. No loops. Just `exec "$@"`.

If your bash has a branch, rewrite it in Haskell.

## Naming Conventions

- **Files**: `lisp-case` (`my-package.nix`, not `myPackage.nix`)
- **Nix attrs**: `lisp-case` (`native-build-inputs`, not `nativeBuildInputs`)
- **Haskell modules**: `PascalCase` (`Weyl.Script.Tools.Rg`)
- **Haskell functions**: `camelCase` with trailing underscore for options (`verbose_`, `output_`)

The membrane translates at boundaries. Inside aleph-naught, everything is `lisp-case`.

## Documentation

- **RFCs go in**: `docs/languages/nix/rfc/aleph-NNN-title.md`
- **Guides go in**: `docs/languages/nix/guides/`
- **Reference goes in**: `docs/languages/nix/reference/`

Use ℵ symbol in display text, `aleph-` in filenames:

```markdown
[ℵ-001](aleph-001-standard-nix.md)  ✓
[ℵ-001](ℵ-001-standard-nix.md)      ✗ (Unicode in filenames)
```

## Testing

```bash
# Run checks
nix flake check

# Build a specific package
nix build .#libmodern.fmt

# Test a Haskell script
nix run .#my-script -- args

# Lint Nix
nix run .#lint
```

## Commit Messages

```
component: short description

Longer explanation if needed.

Refs: ℵ-004
```

Components: `libmodern`, `typed-unix`, `prelude`, `modules`, `docs`, `scripts`

## Pull Requests

1. One logical change per PR
1. Reference the relevant RFC if applicable
1. Update PROGRESS.md if completing a milestone
1. Ensure `nix flake check` passes

## Questions?

Open an issue or check the RFCs:

- [ℵ-001](docs/languages/nix/rfc/aleph-001-standard-nix.md) — Code style
- [ℵ-004](docs/languages/nix/rfc/aleph-004-typed-unix.md) — Typed Unix architecture
- [ℵ-006](docs/languages/nix/rfc/aleph-006-safe-bash.md) — What counts as "safe" bash
