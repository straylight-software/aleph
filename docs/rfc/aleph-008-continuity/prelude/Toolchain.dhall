--| Toolchain Definitions
--|
--| compiler + target + flags = toolchain
--| toolchain + sources + deps = build
--| build -> artifact -> hash

let Types = ./Types.dhall

-- =============================================================================
-- Compiler Kinds
-- =============================================================================

let ClangVersion = { major : Natural, minor : Natural, patch : Natural }
let RustcVersion = { major : Natural, minor : Natural, patch : Natural }
let GHCVersion = { major : Natural, minor : Natural, patch : Natural }
let LeanVersion = { major : Natural, minor : Natural, patch : Natural }
let PursVersion = { major : Natural, minor : Natural, patch : Natural }

let Compiler =
      < Clang : { version : ClangVersion, artifact : Types.Artifact }
      | Rustc : { version : RustcVersion, artifact : Types.Artifact }
      | GHC : { version : GHCVersion, artifact : Types.Artifact }
      | Lean : { version : LeanVersion, artifact : Types.Artifact }
      | Purs : { version : PursVersion, artifact : Types.Artifact }
      | NvLLVM : { version : ClangVersion, artifact : Types.Artifact }  -- Your CUDA LLVM
      >

-- =============================================================================
-- Linkers
-- =============================================================================

let Linker =
      < LLD : Types.Artifact
      | Mold : Types.Artifact
      | Gold : Types.Artifact
      | BFD : Types.Artifact
      | System
      >

-- =============================================================================
-- Toolchain Definition
-- =============================================================================

let Toolchain =
      { compiler : Compiler
      , host : Types.Triple
      , target : Types.Triple
      , flags : List Types.Flag
      , linker : Linker
      , sysroot : Optional Types.Artifact
      , libc : Optional Types.Artifact
      , libcxx : Optional Types.Artifact
      }

-- =============================================================================
-- Standard Triples
-- =============================================================================

let x86_64_linux : Types.Triple =
      { arch = Types.Arch.x86_64
      , vendor = Types.Vendor.unknown
      , os = Types.OS.linux
      , abi = Types.ABI.gnu
      }

let aarch64_linux : Types.Triple =
      { arch = Types.Arch.aarch64
      , vendor = Types.Vendor.unknown
      , os = Types.OS.linux
      , abi = Types.ABI.gnu
      }

let wasm32_wasi : Types.Triple =
      { arch = Types.Arch.wasm32
      , vendor = Types.Vendor.unknown
      , os = Types.OS.wasi
      , abi = Types.ABI.unknown
      }

let orin : Types.Triple =
      { arch = Types.Arch.aarch64
      , vendor = Types.Vendor.nvidia
      , os = Types.OS.linux
      , abi = Types.ABI.gnu
      }

-- =============================================================================
-- Toolchain Constructors
-- =============================================================================

let mkClangToolchain =
      \(version : ClangVersion) ->
      \(artifact : Types.Artifact) ->
      \(target : Types.Triple) ->
      \(flags : List Types.Flag) ->
        { compiler = Compiler.Clang { version, artifact }
        , host = x86_64_linux
        , target
        , flags
        , linker = Linker.System
        , sysroot = None Types.Artifact
        , libc = None Types.Artifact
        , libcxx = None Types.Artifact
        }

let mkRustToolchain =
      \(version : RustcVersion) ->
      \(artifact : Types.Artifact) ->
      \(target : Types.Triple) ->
      \(flags : List Types.Flag) ->
        { compiler = Compiler.Rustc { version, artifact }
        , host = x86_64_linux
        , target
        , flags
        , linker = Linker.System
        , sysroot = None Types.Artifact
        , libc = None Types.Artifact
        , libcxx = None Types.Artifact
        }

let mkGHCToolchain =
      \(version : GHCVersion) ->
      \(artifact : Types.Artifact) ->
      \(target : Types.Triple) ->
      \(flags : List Types.Flag) ->
        { compiler = Compiler.GHC { version, artifact }
        , host = x86_64_linux
        , target
        , flags
        , linker = Linker.System
        , sysroot = None Types.Artifact
        , libc = None Types.Artifact
        , libcxx = None Types.Artifact
        }

let mkLeanToolchain =
      \(version : LeanVersion) ->
      \(artifact : Types.Artifact) ->
        { compiler = Compiler.Lean { version, artifact }
        , host = x86_64_linux
        , target = x86_64_linux
        , flags = [] : List Types.Flag
        , linker = Linker.System
        , sysroot = None Types.Artifact
        , libc = None Types.Artifact
        , libcxx = None Types.Artifact
        }

let mkNvToolchain =
      \(version : ClangVersion) ->
      \(artifact : Types.Artifact) ->
      \(sm : Types.Cpu) ->
        { compiler = Compiler.NvLLVM { version, artifact }
        , host = x86_64_linux
        , target = x86_64_linux  -- Host-side
        , flags = [ Types.Flag.TargetCpu sm ]
        , linker = Linker.System
        , sysroot = None Types.Artifact
        , libc = None Types.Artifact
        , libcxx = None Types.Artifact
        }

-- =============================================================================
-- Exports
-- =============================================================================

in  { Compiler
    , ClangVersion
    , RustcVersion
    , GHCVersion
    , LeanVersion
    , PursVersion
    , Linker
    , Toolchain
    , x86_64_linux
    , aarch64_linux
    , wasm32_wasi
    , orin
    , mkClangToolchain
    , mkRustToolchain
    , mkGHCToolchain
    , mkLeanToolchain
    , mkNvToolchain
    }
