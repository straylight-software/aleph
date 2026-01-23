--| Toolchain Types
--|
--| compiler + target + flags = toolchain
--| That's it. The rest is ceremony.

let Target = ./Target.dhall

let Hash =
      { sha256 : Text
      }

let Artifact =
      { hash : Hash
      , name : Text
      }

let OptLevel =
      < O0 | O1 | O2 | O3 | Oz | Os >

let LTOMode =
      < Off | Thin | Fat >

let DebugInfo =
      < None | LineTablesOnly | Full >

let PanicStrategy =
      < Unwind | Abort >

let Flag =
      < OptLevel : OptLevel
      | LTO : LTOMode
      | Debug : DebugInfo
      | Panic : PanicStrategy
      | TargetCpu : Target.Cpu
      | PIC : Bool
      | RelocationModel : Text
      | CodeModel : Text
      | Feature : { enable : Bool, name : Text }
      | Define : { name : Text, value : Optional Text }
      | Include : Text
      | LibPath : Text
      | Link : Text
      | Raw : Text  -- escape hatch, logged + warned
      >

let CompilerKind =
      < Clang : { version : Text }
      | GCC : { version : Text }
      | Rustc : { version : Text }
      | GHC : { version : Text }
      | Lean : { version : Text }
      >

let Compiler =
      { kind : CompilerKind
      , artifact : Artifact
      }

let Linker =
      < LLD
      | Gold
      | BFD
      | Mold
      | System
      >

let Toolchain =
      { compiler : Compiler
      , host : Target.Triple
      , target : Target.Triple
      , flags : List Flag
      , linker : Optional Linker
      , sysroot : Optional Artifact
      }

-- Smart constructors

let clang
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.Clang { version }, artifact }

let rustc
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.Rustc { version }, artifact }

let ghc
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.GHC { version }, artifact }

let lean
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.Lean { version }, artifact }

let nativeToolchain
    : Compiler -> List Flag -> Toolchain
    = \(compiler : Compiler) ->
      \(flags : List Flag) ->
        { compiler
        , host = Target.x86_64_linux
        , target = Target.x86_64_linux
        , flags
        , linker = Some Linker.LLD
        , sysroot = None Artifact
        }

let crossToolchain
    : Compiler -> Target.Triple -> Artifact -> List Flag -> Toolchain
    = \(compiler : Compiler) ->
      \(target : Target.Triple) ->
      \(sysroot : Artifact) ->
      \(flags : List Flag) ->
        { compiler
        , host = Target.x86_64_linux
        , target
        , flags
        , linker = Some Linker.LLD
        , sysroot = Some sysroot
        }

in  { Hash
    , Artifact
    , OptLevel
    , LTOMode
    , DebugInfo
    , PanicStrategy
    , Flag
    , CompilerKind
    , Compiler
    , Linker
    , Toolchain
    , clang
    , rustc
    , ghc
    , lean
    , nativeToolchain
    , crossToolchain
    }
