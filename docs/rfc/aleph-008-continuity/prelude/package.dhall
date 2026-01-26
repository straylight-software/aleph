--| Straylight Prelude
--|
--| The typed interface for builds.
--| Import this to define BUILD.dhall files.
--|
--| Usage:
--|   let S = https://straylight.cx/prelude/v1/package.dhall sha256:...
--|   in [ S.rust_library "mylib" ["src/lib.rs"] [] ]

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall
let Rust = ./Rust.dhall
let Lean = ./Lean.dhall
let Cxx = ./Cxx.dhall
let Haskell = ./Haskell.dhall
let PureScript = ./PureScript.dhall
let Nv = ./Nv.dhall

in  {
    -- ==========================================================================
    -- Types
    -- ==========================================================================
      Hash = Types.Hash
    , Artifact = Types.Artifact
    , Arch = Types.Arch
    , OS = Types.OS
    , ABI = Types.ABI
    , Vendor = Types.Vendor
    , Cpu = Types.Cpu
    , Triple = Types.Triple
    , OptLevel = Types.OptLevel
    , LTOMode = Types.LTOMode
    , DebugInfo = Types.DebugInfo
    , Flag = Types.Flag
    , Visibility = Types.Visibility
    , Dep = Types.Dep

    -- ==========================================================================
    -- Toolchains
    -- ==========================================================================
    , Compiler = Toolchain.Compiler
    , Linker = Toolchain.Linker
    , Toolchain = Toolchain.Toolchain
    , x86_64_linux = Toolchain.x86_64_linux
    , aarch64_linux = Toolchain.aarch64_linux
    , wasm32_wasi = Toolchain.wasm32_wasi
    , orin = Toolchain.orin
    , mkClangToolchain = Toolchain.mkClangToolchain
    , mkRustToolchain = Toolchain.mkRustToolchain
    , mkGHCToolchain = Toolchain.mkGHCToolchain
    , mkLeanToolchain = Toolchain.mkLeanToolchain
    , mkNvToolchain = Toolchain.mkNvToolchain

    -- ==========================================================================
    -- Rust
    -- ==========================================================================
    , RustEdition = Rust.Edition
    , CrateType = Rust.CrateType
    , RustLibrary = Rust.RustLibrary
    , rust_library = Rust.rust_library
    , RustBinary = Rust.RustBinary
    , rust_binary = Rust.rust_binary
    , RustTest = Rust.RustTest
    , rust_test = Rust.rust_test

    -- ==========================================================================
    -- Lean4
    -- ==========================================================================
    , LeanBackend = Lean.Backend
    , LeanLibrary = Lean.LeanLibrary
    , lean_library = Lean.lean_library
    , LeanBinary = Lean.LeanBinary
    , lean_binary = Lean.lean_binary
    , LeanTest = Lean.LeanTest
    , lean_test = Lean.lean_test
    , ProvenLibrary = Lean.ProvenLibrary
    , proven_library = Lean.proven_library

    -- ==========================================================================
    -- C/C++
    -- ==========================================================================
    , CxxStandard = Cxx.CxxStandard
    , CStandard = Cxx.CStandard
    , CxxLibrary = Cxx.CxxLibrary
    , cxx_library = Cxx.cxx_library
    , CxxBinary = Cxx.CxxBinary
    , cxx_binary = Cxx.cxx_binary
    , CxxTest = Cxx.CxxTest
    , cxx_test = Cxx.cxx_test

    -- ==========================================================================
    -- Haskell
    -- ==========================================================================
    , HaskellExtension = Haskell.Extension
    , HaskellLibrary = Haskell.HaskellLibrary
    , haskell_library = Haskell.haskell_library
    , HaskellBinary = Haskell.HaskellBinary
    , haskell_binary = Haskell.haskell_binary
    , HaskellTest = Haskell.HaskellTest
    , haskell_test = Haskell.haskell_test

    -- ==========================================================================
    -- PureScript
    -- ==========================================================================
    , PureScriptBackend = PureScript.Backend
    , PureScriptBundler = PureScript.Bundler
    , PureScriptLibrary = PureScript.PureScriptLibrary
    , purescript_library = PureScript.purescript_library
    , PureScriptBundle = PureScript.PureScriptBundle
    , purescript_bundle = PureScript.purescript_bundle
    , PureScriptTest = PureScript.PureScriptTest
    , purescript_test = PureScript.purescript_test

    -- ==========================================================================
    -- nv (CUDA)
    -- ==========================================================================
    , SmArch = Nv.SmArch
    , CudaStandard = Nv.CudaStandard
    , NvOutputKind = Nv.OutputKind
    , NvLibrary = Nv.NvLibrary
    , nv_library = Nv.nv_library
    , NvBinary = Nv.NvBinary
    , nv_binary = Nv.nv_binary
    , NvTest = Nv.NvTest
    , nv_test = Nv.nv_test

    -- ==========================================================================
    -- Target helpers
    -- ==========================================================================
    , target = \(rule : Text) -> \(attrs : Type) -> \(a : attrs) ->
        { rule, attrs = a }
    }
