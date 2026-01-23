--| Core Types for Straylight Prelude
--|
--| These are the foundational types shared across all rules.
--| Content-addressed imports. No strings where enums suffice.

-- =============================================================================
-- Hashes and Artifacts
-- =============================================================================

let Hash = { sha256 : Text }

let Artifact =
      { hash : Hash
      , name : Text
      }

-- =============================================================================
-- Target Triples
-- =============================================================================

let Arch = < x86_64 | aarch64 | wasm32 | riscv64 >

let OS = < linux | darwin | wasi | none >

let ABI = < gnu | musl | eabi | unknown >

let Vendor = < unknown | apple | pc | nvidia >

let Cpu =
      < generic
      | native
      -- x86_64
      | znver3
      | znver4
      | skylake
      | sapphirerapids
      -- aarch64
      | cortex_a78ae  -- Orin
      | neoverse_n1   -- Graviton 2
      | neoverse_v1   -- Graviton 3
      | apple_m1
      | apple_m2
      -- GPU
      | sm_89         -- Ada Lovelace (RTX 40xx)
      | sm_90         -- Hopper (H100)
      | sm_120        -- Blackwell
      >

let Triple =
      { arch : Arch
      , vendor : Vendor
      , os : OS
      , abi : ABI
      }

-- =============================================================================
-- Compiler Flags (Typed, not strings)
-- =============================================================================

let OptLevel = < O0 | O1 | O2 | O3 | Oz | Os >

let LTOMode = < Off | Thin | Fat >

let DebugInfo = < None | LineTablesOnly | Full >

let PanicStrategy = < Unwind | Abort >

let Sanitizer = < Address | Thread | Memory | Undefined | Leak >

let Flag =
      < OptLevel : OptLevel
      | LTO : LTOMode
      | Debug : DebugInfo
      | Panic : PanicStrategy
      | Sanitizer : Sanitizer
      | TargetCpu : Cpu
      | TargetFeature : { enable : Bool, name : Text }
      | PIC : Bool
      | Define : { name : Text, value : Optional Text }
      | Include : Text
      | LibPath : Text
      | Link : Text
      | Std : Text           -- C/C++ standard
      | Warnings : < All | None | Error >
      | Raw : Text           -- Escape hatch, logged
      >

-- =============================================================================
-- Visibility
-- =============================================================================

let Visibility =
      < Public
      | Private
      | Package
      | Targets : List Text
      >

-- =============================================================================
-- Common Target Fields
-- =============================================================================

let CommonAttrs =
      { name : Text
      , visibility : Visibility
      , labels : List Text
      , licenses : List Text
      }

let defaultCommon =
      \(name : Text) ->
        { name
        , visibility = Visibility.Public
        , labels = [] : List Text
        , licenses = [] : List Text
        }

-- =============================================================================
-- Source Files (NO GLOBS)
-- =============================================================================

let Sources =
      { srcs : List Text
      , hdrs : Optional (List Text)  -- For C/C++
      }

-- =============================================================================
-- Dependencies
-- =============================================================================

let Dep =
      < Target : Text           -- "//path:target" or ":local"
      | External : Artifact     -- Content-addressed external dep
      >

-- =============================================================================
-- Exports
-- =============================================================================

in  { Hash
    , Artifact
    , Arch
    , OS
    , ABI
    , Vendor
    , Cpu
    , Triple
    , OptLevel
    , LTOMode
    , DebugInfo
    , PanicStrategy
    , Sanitizer
    , Flag
    , Visibility
    , CommonAttrs
    , defaultCommon
    , Sources
    , Dep
    }
