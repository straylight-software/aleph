--| Toolchain Types
--|
--| compiler + target + flags = toolchain
--| That's it. The rest is ceremony.

let Target = ./Target.dhall

--------------------------------------------------------------------------------
-- Artifacts (content-addressed)
--------------------------------------------------------------------------------

let Hash =
      { sha256 : Text
      }

let Artifact =
      { hash : Hash
      , name : Text
      }

--------------------------------------------------------------------------------
-- Compiler Flags (typed, not strings)
--------------------------------------------------------------------------------

let OptLevel =
      < O0 | O1 | O2 | O3 | Oz | Os >

let LTOMode =
      < off | thin | fat >

let DebugInfo =
      < none | line-tables | full >

let Flag =
      < opt-level : OptLevel
      | lto : LTOMode
      | debug : DebugInfo
      | target-cpu : Target.Cpu
      | target-gpu : Target.Gpu
      | pic : Bool
      | define : { name : Text, value : Optional Text }
      | include : Text
      | lib-path : Text
      | link : Text
      | std : Text           -- C/C++ standard (c23, c++23)
      | raw : Text           -- escape hatch, logged + warned
      >

--------------------------------------------------------------------------------
-- Compilers
--------------------------------------------------------------------------------

let CompilerKind =
      -- C/C++
      < nv-clang : { version : Text }    -- LLVM git with CUDA C++23 patches
      | clang : { version : Text }       -- stock LLVM/Clang
      | gcc : { version : Text }         -- GCC
      -- Rust
      | rustc : { version : Text }
      -- Haskell
      | ghc : { version : Text }
      -- Lean
      | lean : { version : Text }
      -- Python (for completeness)
      | python : { version : Text }
      >

let Compiler =
      { kind : CompilerKind
      , artifact : Artifact
      }

--------------------------------------------------------------------------------
-- Linkers
--------------------------------------------------------------------------------

let Linker =
      < lld
      | gold
      | bfd
      | mold
      | system
      >

--------------------------------------------------------------------------------
-- C++ Standard Library
--------------------------------------------------------------------------------

let CxxStdlib =
      < libcxx            -- LLVM libc++
      | libstdcxx         -- GNU libstdc++
      >

--------------------------------------------------------------------------------
-- Toolchain
--------------------------------------------------------------------------------

let Toolchain =
      { compiler : Compiler
      , host : Target.Triple
      , target : Target.Triple
      , flags : List Flag
      , linker : Optional Linker
      , cxx-stdlib : Optional CxxStdlib
      , sysroot : Optional Artifact
      }

--------------------------------------------------------------------------------
-- Smart constructors
--------------------------------------------------------------------------------

let nv-clang
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.nv-clang { version }, artifact }

let clang
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.clang { version }, artifact }

let gcc
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.gcc { version }, artifact }

let rustc
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.rustc { version }, artifact }

let ghc
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.ghc { version }, artifact }

let lean
    : Text -> Artifact -> Compiler
    = \(version : Text) ->
      \(artifact : Artifact) ->
        { kind = CompilerKind.lean { version }, artifact }

--------------------------------------------------------------------------------
-- Common toolchain patterns
--------------------------------------------------------------------------------

let native-toolchain
    : Compiler -> CxxStdlib -> List Flag -> Toolchain
    = \(compiler : Compiler) ->
      \(stdlib : CxxStdlib) ->
      \(flags : List Flag) ->
        { compiler
        , host = Target.x86-64-linux
        , target = Target.x86-64-linux
        , flags
        , linker = Some Linker.lld
        , cxx-stdlib = Some stdlib
        , sysroot = None Artifact
        }

let cross-toolchain
    : Compiler -> Target.Triple -> CxxStdlib -> Artifact -> List Flag -> Toolchain
    = \(compiler : Compiler) ->
      \(target : Target.Triple) ->
      \(stdlib : CxxStdlib) ->
      \(sysroot : Artifact) ->
      \(flags : List Flag) ->
        { compiler
        , host = Target.x86-64-linux
        , target
        , flags
        , linker = Some Linker.lld
        , cxx-stdlib = Some stdlib
        , sysroot = Some sysroot
        }

in  { -- Types
      Hash
    , Artifact
    , OptLevel
    , LTOMode
    , DebugInfo
    , Flag
    , CompilerKind
    , Compiler
    , Linker
    , CxxStdlib
    , Toolchain
      -- Constructors
    , nv-clang
    , clang
    , gcc
    , rustc
    , ghc
    , lean
    , native-toolchain
    , cross-toolchain
    }
