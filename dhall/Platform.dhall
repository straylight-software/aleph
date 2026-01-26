-- dhall/Platform.dhall
--
-- Platform and toolchain configuration.

let Path = { _path : Text }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Platform
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let Cpu = < X86_64 | Aarch64 >
let Os = < Linux | Darwin >

let Platform = { cpu : Cpu, os : Os }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Toolchain paths (all from Nix store)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let CxxToolchain =
    { clang : Path
    , clangxx : Path
    , lld : Path
    , ar : Path
    , resourceDir : Path
    , gccInclude : Path
    , gccIncludeArch : Path
    , glibcInclude : Path
    , gccLib : Path
    , gccLibBase : Path
    , glibcLib : Path
    }

let NvToolchain =
    { cxx : CxxToolchain          -- inherits C++ toolchain
    , nvidiaSdkInclude : Path
    , nvidiaSdkLib : Path
    }

let HaskellToolchain =
    { ghc : Path
    , ghcPkg : Path
    , haddock : Path
    }

let RustToolchain =
    { rustc : Path
    , cargo : Path
    , clippy : Path
    }

let LeanToolchain =
    { lean : Path
    , lake : Path
    }

let PythonToolchain =
    { python : Path
    }

let Toolchain =
    { cxx : CxxToolchain
    , nv : NvToolchain
    , haskell : HaskellToolchain
    , rust : RustToolchain
    , lean : LeanToolchain
    , python : PythonToolchain
    }

in  { Cpu, Os, Platform
    , CxxToolchain, NvToolchain, HaskellToolchain, RustToolchain, LeanToolchain, PythonToolchain
    , Toolchain
    }
