--| nv (CUDA) Rules
--|
--| Your hacked LLVM with C++23 for CUDA
--| nv_library, nv_binary

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- CUDA-specific Types
-- =============================================================================

let SmArch =
      < SM_50   -- Maxwell
      | SM_60   -- Pascal
      | SM_70   -- Volta
      | SM_75   -- Turing
      | SM_80   -- Ampere
      | SM_86   -- Ampere (GA10x)
      | SM_89   -- Ada Lovelace
      | SM_90   -- Hopper
      | SM_120  -- Blackwell
      >

let CudaStandard = < Cuda11 | Cuda12 >

let OutputKind =
      < PTX     -- NVIDIA PTX assembly
      | Cubin   -- CUDA binary
      | Fatbin  -- Multi-arch fat binary
      | Object  -- Host object with embedded device code
      >

-- =============================================================================
-- nv_library
-- =============================================================================

let NvLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text           -- .cu files
      , hdrs : List Text           -- .cuh files
      , deps : List Types.Dep
      , sm_archs : List SmArch     -- Target GPU architectures
      , cxx_std : < Cxx17 | Cxx20 | Cxx23 >
      , cuda_std : CudaStandard
      , defines : List { name : Text, value : Optional Text }
      , nvflags : List Text
      , output : OutputKind
      , toolchain : Optional Toolchain.Toolchain
      }

let nv_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(sm_archs : List SmArch) ->
        { common = Types.defaultCommon name
        , srcs
        , hdrs = [] : List Text
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , sm_archs
        , cxx_std = < Cxx17 | Cxx20 | Cxx23 >.Cxx23
        , cuda_std = CudaStandard.Cuda12
        , defines = [] : List { name : Text, value : Optional Text }
        , nvflags = [] : List Text
        , output = OutputKind.Object
        , toolchain = None Toolchain.Toolchain
        } : NvLibrary

-- =============================================================================
-- nv_binary
-- =============================================================================

let NvBinary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , sm_archs : List SmArch
      , cxx_std : < Cxx17 | Cxx20 | Cxx23 >
      , cuda_std : CudaStandard
      , defines : List { name : Text, value : Optional Text }
      , nvflags : List Text
      , link_style : < Static | Shared >
      , toolchain : Optional Toolchain.Toolchain
      }

let nv_binary =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(sm_archs : List SmArch) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , sm_archs
        , cxx_std = < Cxx17 | Cxx20 | Cxx23 >.Cxx23
        , cuda_std = CudaStandard.Cuda12
        , defines = [] : List { name : Text, value : Optional Text }
        , nvflags = [] : List Text
        , link_style = < Static | Shared >.Static
        , toolchain = None Toolchain.Toolchain
        } : NvBinary

-- =============================================================================
-- nv_test
-- =============================================================================

let NvTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , sm_archs : List SmArch
      , toolchain : Optional Toolchain.Toolchain
      }

let nv_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(sm_archs : List SmArch) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , sm_archs
        , toolchain = None Toolchain.Toolchain
        } : NvTest

-- =============================================================================
-- Exports
-- =============================================================================

in  { SmArch
    , CudaStandard
    , OutputKind
    , NvLibrary
    , nv_library
    , NvBinary
    , nv_binary
    , NvTest
    , nv_test
    }
