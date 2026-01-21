-- Aleph/Config/CUDA.dhall
--
-- CUDA-specific types for S4 inference builds

let Drv = ./Drv.dhall

-- ============================================================================
-- GPU Architectures
-- ============================================================================

let Arch =
  < SM_70   -- Volta (V100)
  | SM_75   -- Turing (RTX 20xx, T4)
  | SM_80   -- Ampere (A100)
  | SM_86   -- Ampere (RTX 30xx, A10)
  | SM_89   -- Ada (RTX 40xx, L40)
  | SM_90   -- Hopper (H100)
  | SM_90a  -- Hopper + wgmma
  | SM_100  -- Blackwell (B100, B200)
  | SM_120  -- Blackwell Ultra
  >

let archToFlag : Arch → Text =
  λ(a : Arch) →
    merge
      { SM_70 = "70"
      , SM_75 = "75"
      , SM_80 = "80"
      , SM_86 = "86"
      , SM_89 = "89"
      , SM_90 = "90"
      , SM_90a = "90a"
      , SM_100 = "100"
      , SM_120 = "120"
      }
      a

let archToGencode : Arch → Text =
  λ(a : Arch) →
    let flag = archToFlag a
    in "-gencode=arch=compute_${flag},code=sm_${flag}"

-- ============================================================================
-- Quantization
-- ============================================================================

let Quantization =
  < BF16
  | FP16
  | FP8_E4M3
  | FP8_E5M2
  | NVFP4
  | INT8
  | INT4
  >

let quantToDefine : Quantization → Text =
  λ(q : Quantization) →
    merge
      { BF16 = "-DUSE_BF16"
      , FP16 = "-DUSE_FP16"
      , FP8_E4M3 = "-DUSE_FP8 -DFP8_FORMAT=E4M3"
      , FP8_E5M2 = "-DUSE_FP8 -DFP8_FORMAT=E5M2"
      , NVFP4 = "-DUSE_NVFP4"
      , INT8 = "-DUSE_INT8"
      , INT4 = "-DUSE_INT4"
      }
      q

-- ============================================================================
-- Model Formats
-- ============================================================================

let ModelFormat =
  < Safetensors
  | GGUF
  | PyTorch
  | ONNX
  >

-- ============================================================================
-- CUDA Build Actions
-- ============================================================================

-- | NVCC compilation
let Nvcc =
  { srcs : List Drv.Ref
  , out : Drv.Ref
  , arch : List Arch
  , optLevel : Natural     -- -O level
  , fastMath : Bool
  , debug : Bool
  , lineInfo : Bool
  , ptx : Bool             -- emit PTX
  , fatbin : Bool          -- emit fatbin
  , includes : List Drv.Ref
  , libs : List Drv.Ref
  , defines : List { key : Text, value : Optional Text }
  , extraFlags : List Text
  }

let defaultNvcc : Nvcc =
  { srcs = [] : List Drv.Ref
  , out = Drv.rel "a.out"
  , arch = [Arch.SM_90]
  , optLevel = 3
  , fastMath = True
  , debug = False
  , lineInfo = False
  , ptx = False
  , fatbin = False
  , includes = [] : List Drv.Ref
  , libs = [] : List Drv.Ref
  , defines = [] : List { key : Text, value : Optional Text }
  , extraFlags = [] : List Text
  }

-- | Convert Nvcc to a Tool action
let nvccToAction : Nvcc → Drv.Action =
  λ(opts : Nvcc) →
    let archFlags = Prelude.List.map Arch Text archToGencode opts.arch
    let optFlag = "-O${Natural/show opts.optLevel}"
    let fastMathFlag = if opts.fastMath then ["--use_fast_math"] else [] : List Text
    let debugFlags = 
        (if opts.debug then ["-g"] else [] : List Text) #
        (if opts.lineInfo then ["-lineinfo"] else [] : List Text)
    let formatFlags =
        (if opts.ptx then ["--ptx"] else [] : List Text) #
        (if opts.fatbin then ["--fatbin"] else [] : List Text)
    let includeFlags = Prelude.List.concatMap Drv.Ref Text
        (λ(i : Drv.Ref) → ["-I", Drv.Expr.Ref i]) 
        opts.includes
    -- ... full implementation would be longer
    in Drv.tool "cuda" "nvcc" 
        (Prelude.List.map Text Drv.Expr (λ(t : Text) → Drv.Expr.Str t)
          (archFlags # [optFlag] # fastMathFlag # debugFlags # formatFlags # opts.extraFlags))

-- ============================================================================
-- CUTLASS Build
-- ============================================================================

let CutlassOpts =
  { arch : Arch
  , operations : List Text   -- gemm, conv, etc
  , tileM : Natural
  , tileN : Natural
  , tileK : Natural
  }

-- ============================================================================
-- S4 Build Inputs
-- ============================================================================

let S4Inputs =
  { modelName : Text
  , modelFormat : ModelFormat
  , quantization : Quantization
  , targetArch : List Arch
  , kernelSrcs : List Text
  , extraFlags : List Text
  }

let defaultS4Inputs : S4Inputs =
  { modelName = "unnamed"
  , modelFormat = ModelFormat.Safetensors
  , quantization = Quantization.BF16
  , targetArch = [Arch.SM_90]
  , kernelSrcs = [] : List Text
  , extraFlags = [] : List Text
  }

-- Presets
let blackwellNVFP4 : Text → List Text → S4Inputs =
  λ(name : Text) → λ(srcs : List Text) →
    defaultS4Inputs //
      { modelName = name
      , quantization = Quantization.NVFP4
      , targetArch = [Arch.SM_100]
      , kernelSrcs = srcs
      }

let hopperFP8 : Text → List Text → S4Inputs =
  λ(name : Text) → λ(srcs : List Text) →
    defaultS4Inputs //
      { modelName = name
      , quantization = Quantization.FP8_E4M3
      , targetArch = [Arch.SM_90, Arch.SM_90a]
      , kernelSrcs = srcs
      }

let adaBF16 : Text → List Text → S4Inputs =
  λ(name : Text) → λ(srcs : List Text) →
    defaultS4Inputs //
      { modelName = name
      , quantization = Quantization.BF16
      , targetArch = [Arch.SM_89]
      , kernelSrcs = srcs
      }

-- ============================================================================
-- Exports
-- ============================================================================

in  { -- Architecture
      Arch
    , archToFlag
    , archToGencode
    
    -- Quantization
    , Quantization
    , quantToDefine
    
    -- Model formats
    , ModelFormat
    
    -- CUDA build
    , Nvcc
    , defaultNvcc
    , nvccToAction
    
    -- CUTLASS
    , CutlassOpts
    
    -- S4
    , S4Inputs
    , defaultS4Inputs
    , blackwellNVFP4
    , hopperFP8
    , adaBF16
    }
