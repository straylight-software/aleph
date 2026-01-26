--| Continuity Build System - Dhall Package
--|
--| The typed interface to prelude-less Buck2 (DICE).
--| Import this to define builds.
--|
--| Usage:
--|   let DICE = https://straylight.cx/dice/v1/package.dhall sha256:...
--|   in DICE.rust_library "mylib" ["src/lib.rs"] []

let Target = ./Target.dhall
let Toolchain = ./Toolchain.dhall
let Action = ./Action.dhall
let Rules = ./Rules.dhall

in  { -- Target types
      Arch = Target.Arch
    , OS = Target.OS
    , ABI = Target.ABI
    , Cpu = Target.Cpu
    , Triple = Target.Triple
    , tripleToString = Target.tripleToString
    
    -- Common targets
    , x86_64_linux = Target.x86_64_linux
    , aarch64_linux = Target.aarch64_linux
    , wasm32_wasi = Target.wasm32_wasi
    , orin = Target.orin
    
    -- Toolchain types
    , Hash = Toolchain.Hash
    , Artifact = Toolchain.Artifact
    , OptLevel = Toolchain.OptLevel
    , LTOMode = Toolchain.LTOMode
    , DebugInfo = Toolchain.DebugInfo
    , Flag = Toolchain.Flag
    , Compiler = Toolchain.Compiler
    , CompilerKind = Toolchain.CompilerKind
    , Linker = Toolchain.Linker
    , Toolchain = Toolchain.Toolchain
    
    -- Toolchain constructors
    , clang = Toolchain.clang
    , rustc = Toolchain.rustc
    , ghc = Toolchain.ghc
    , lean = Toolchain.lean
    , nativeToolchain = Toolchain.nativeToolchain
    , crossToolchain = Toolchain.crossToolchain
    
    -- Action types
    , ActionCategory = Action.ActionCategory
    , EnvVar = Action.EnvVar
    , Input = Action.Input
    , Output = Action.Output
    , Action = Action.Action
    
    -- Action constructors
    , compile = Action.compile
    , link = Action.link
    , copy = Action.copy
    , write = Action.write
    , run = Action.run
    
    -- Rule types
    , Visibility = Rules.Visibility
    , RustEdition = Rules.RustEdition
    , CrateType = Rules.CrateType
    , RustLibrary = Rules.RustLibrary
    , RustBinary = Rules.RustBinary
    , LeanLibrary = Rules.LeanLibrary
    , CxxStandard = Rules.CxxStandard
    , CLibrary = Rules.CLibrary
    , WasmOptLevel = Rules.WasmOptLevel
    , WasmFeature = Rules.WasmFeature
    , WasmModule = Rules.WasmModule
    , Rule = Rules.Rule
    
    -- Rule constructors
    , rust_library = Rules.rust_library
    , rust_binary = Rules.rust_binary
    , lean_library = Rules.lean_library
    , c_library = Rules.c_library
    , wasm_module = Rules.wasm_module
    }
