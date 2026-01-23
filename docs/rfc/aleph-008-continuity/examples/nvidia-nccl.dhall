-- | nvidia-nccl: NVIDIA NCCL Multi-GPU Communication Library
--|
--| This is the reference example showing how to express a typed package
--| definition in Dhall that compiles down to either:
--|
--| 1. Nix derivation (via Derivation.dhall -> dhall-to-nix)
--| 2. Haskell Action list (via Script -> Action bridge)  
--| 3. DICE actions (via Script -> WASM/Starlark)
--|
--| The key insight: the Script.dhall Command type is the PORTABLE representation.
--| Everything else is a compilation target.

let Script = ../prelude/Script.dhall
let Types = ../prelude/Types.dhall

-- =============================================================================
-- Package Metadata
-- =============================================================================

let pname = "nvidia-nccl"
let version = "2.28.9"

let src =
      { url = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl"
      , sha256 = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
      }

-- =============================================================================
-- Build Script (Typed Commands)
-- =============================================================================

-- | The install phase expressed as typed commands.
-- No string interpolation bugs. No quoting errors. Auditable.
let installScript : Script.Script =
      [ -- Unzip the wheel (it's a zip file)
        Script.Command.Unzip
          { archive = Script.Path.Src ""      -- $src is the wheel file
          , dest = Script.Path.Tmp "unpacked"
          }
      
      -- Create output directories
      , Script.mkdir "lib"
      , Script.mkdir "include"
      
      -- Copy library files
      , Script.Command.Copy
          { src = Script.Path.Tmp "unpacked/nvidia/nccl/lib/."
          , dst = Script.Path.Out "lib/"
          , recursive = True
          }
      
      -- Copy header files
      , Script.Command.Copy
          { src = Script.Path.Tmp "unpacked/nvidia/nccl/include/."
          , dst = Script.Path.Out "include/"
          , recursive = True
          }
      
      -- Create lib64 symlink for compatibility
      , Script.symlink "lib" "lib64"
      ]

-- =============================================================================
-- Dependencies
-- =============================================================================

-- | Native build inputs (build-time tools)
let nativeBuildInputs : List Text =
      [ "autoPatchelfHook"
      , "unzip"
      ]

-- | Build inputs (runtime dependencies for autoPatchelf)
let buildInputs : List Text =
      [ "stdenv.cc.cc.lib"  -- libstdc++
      , "zlib"
      ]

-- =============================================================================
-- Package Definition (the exportable format)
-- =============================================================================

-- | Complete package specification
let NvidiaPackage =
      { Type =
          { pname : Text
          , version : Text
          , src : { url : Text, sha256 : Text }
          , nativeBuildInputs : List Text
          , buildInputs : List Text
          , dontUnpack : Bool
          , installPhase : Script.Script
          , description : Text
          , homepage : Text
          , license : Text
          }
      , default =
          { dontUnpack = False
          , nativeBuildInputs = [] : List Text
          , buildInputs = [] : List Text
          , installPhase = [] : Script.Script
          , description = ""
          , homepage = ""
          , license = "unfree"
          }
      }

let ncclPackage : NvidiaPackage.Type =
      NvidiaPackage::{
      , pname
      , version
      , src
      , nativeBuildInputs
      , buildInputs
      , dontUnpack = True  -- Wheel is unzipped in install phase
      , installPhase = installScript
      , description = "NVIDIA NCCL ${version} (from PyPI)"
      , homepage = "https://developer.nvidia.com/nccl"
      , license = "unfree"
      }

in ncclPackage
