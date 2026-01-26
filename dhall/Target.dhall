-- dhall/Target.dhall
--
-- Build target types for Aleph.

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Primitives (not Text)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let Path = { _path : Text }
let path : Text -> Path = \(t : Text) -> { _path = t }

let Name = { _name : Text }
let name : Text -> Name = \(t : Text) -> { _name = t }

let Label = { package : Optional Text, name : Text }
let label : Text -> Label = \(n : Text) -> { package = None Text, name = n }

let FlakeRef = { flake : Text, attr : Text }
let nixpkgs : Text -> FlakeRef = \(a : Text) -> { flake = "nixpkgs", attr = a }

let Dep = < Local : Label | Nix : FlakeRef >

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Languages and their options
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let CxxStd = < C23 | Cxx17 | Cxx20 | Cxx23 >
let RustEdition = < E2015 | E2018 | E2021 | E2024 >
let CrateType = < Bin | Lib | Rlib | Dylib | Cdylib | Staticlib >
let PyVersion = < Py39 | Py310 | Py311 | Py312 >

-- NVIDIA SM architectures (Blackwell+ only, sm_100 floor)
let SmArch = < SM_100 | SM_120 >

let CxxOpts = { std : CxxStd, cflags : List Text, ldflags : List Text }
let HaskellOpts = { packages : List Text, ghcOptions : List Text, includePaths : List Path }
let RustOpts = { edition : RustEdition, crateType : CrateType, features : List Text }
let LeanOpts = { roots : List Path }
let PythonOpts = { version : PyVersion }

-- NV: CUDA via clang (not nvcc), Blackwell+
let NvOpts =
    { arch : SmArch
    , cxxOpts : CxxOpts  -- inherits C++ settings
    }

-- Language with its opts (a dependent pair, morally)
let Lang =
    < Cxx : CxxOpts
    | Nv : NvOpts
    | Haskell : HaskellOpts
    | Rust : RustOpts
    | Lean : LeanOpts
    | Python : PythonOpts
    >

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- FFI bridges (valid combinations only)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- FFI always has two sides with their own sources and options
let FfiBridge =
    < Haskell_Cxx :
        { hs : { srcs : List Path, opts : HaskellOpts }
        , cxx : { srcs : List Path, opts : CxxOpts }
        }
    | Rust_Cxx :
        { rs : { srcs : List Path, opts : RustOpts }
        , cxx : { srcs : List Path, opts : CxxOpts }
        }
    | Haskell_Rust :
        { hs : { srcs : List Path, opts : HaskellOpts }
        , rs : { srcs : List Path, opts : RustOpts }
        }
    >

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Target kinds
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- What you can build
let TargetKind =
    < Binary
    | Library
    | Test
    >

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Unified target
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let Target =
    < Pure :
        { name : Name
        , kind : TargetKind
        , lang : Lang
        , srcs : List Path
        , deps : List Dep
        }
    | Ffi :
        { name : Name
        , kind : TargetKind
        , bridge : FfiBridge
        , deps : List Dep
        }
    | Prebuilt :
        { name : Name
        , lang : Lang
        , artifact : Path  -- .a, .rlib, whatever
        }
    >

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Defaults
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let defaults =
    { cxx = { std = CxxStd.Cxx23, cflags = [] : List Text, ldflags = [] : List Text }
    , nv = { arch = SmArch.SM_100, cxxOpts = { std = CxxStd.Cxx23, cflags = [] : List Text, ldflags = [] : List Text } }
    , haskell = { packages = [] : List Text, ghcOptions = [] : List Text, includePaths = [] : List Path }
    , rust = { edition = RustEdition.E2021, crateType = CrateType.Bin, features = [] : List Text }
    }

in  { Path, path
    , Name, name
    , Label, label
    , FlakeRef, nixpkgs
    , Dep
    , CxxStd, RustEdition, CrateType, PyVersion, SmArch
    , CxxOpts, HaskellOpts, RustOpts, LeanOpts, PythonOpts, NvOpts
    , Lang
    , FfiBridge
    , TargetKind
    , Target
    , defaults
    }
