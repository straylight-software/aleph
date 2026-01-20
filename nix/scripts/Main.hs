{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

{- | WASM plugin for typed package definitions.

This module exports package definitions that can be called from Nix
via @builtins.wasm@.

= Usage

@
# From Nix (using straylight-nix with builtins.wasm)
let
  wasm = builtins.wasm ./straylight-packages.wasm;
in {
  zlib-ng = wasm "zlib_ng" {};
  fmt = wasm "fmt" {};
}
@

= Building

@
wasm32-wasi-ghc -optl-mexec-model=reactor \
  -package text -package containers \
  Main.hs -o straylight-packages.wasm
@

NOTE: We use 'Main' module name because GHC WASM reactor modules need
proper RTS initialization, which only happens when there's a Main module.
The 'main' function is a dummy that's never called (reactor modules don't
have a _start entry point).
-}
module Main where

import Aleph.Nix
import Aleph.Nix.Derivation
import Aleph.Nix.Packages.AbseilCpp (abseilCpp)
import Aleph.Nix.Packages.Catch2 (catch2)
import Aleph.Nix.Packages.Cutlass (cutlass)
import Aleph.Nix.Packages.Fmt (fmt)
import Aleph.Nix.Packages.HelloWrapped (helloWrapped)
import Aleph.Nix.Packages.Jq (jq)
import Aleph.Nix.Packages.Mdspan (mdspan)
import Aleph.Nix.Packages.NlohmannJson (nlohmannJson)
import Aleph.Nix.Packages.Nvidia (cudnn, cusparselt, cutensor, nccl, tensorrt)
import qualified Aleph.Nix.Packages.Nvidia as Nvidia (cutlass)
import Aleph.Nix.Packages.Rapidjson (rapidjson)
import Aleph.Nix.Packages.Spdlog (spdlog)
import Aleph.Nix.Packages.ZlibNg (zlibNg)

{- | Dummy main for GHC to link RTS properly.
This is never called in reactor mode.
-}
main :: IO ()
main = pure ()

-- | Required initialization function.
foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()

initPlugin :: IO ()
initPlugin = nixWasmInit

{- | Export zlib-ng package specification.

Called from Nix as: @wasm "zlib_ng" {}@

Returns an attrset with the full derivation specification.
-}
foreign export ccall "zlib_ng" zlibNgExport :: Value -> IO Value

zlibNgExport :: Value -> IO Value
zlibNgExport _args = drvToNixAttrs zlibNg

{- | Export fmt package specification.

Called from Nix as: @wasm "fmt" {}@

Returns an attrset with the full derivation specification.
-}
foreign export ccall "fmt" fmtExport :: Value -> IO Value

fmtExport :: Value -> IO Value
fmtExport _args = drvToNixAttrs fmt

{- | Export mdspan package specification.

Called from Nix as: @wasm "mdspan" {}@

Returns an attrset with the full derivation specification.
Includes typed postInstall phase to write C++23 shim header.
-}
foreign export ccall "mdspan" mdspanExport :: Value -> IO Value

mdspanExport :: Value -> IO Value
mdspanExport _args = drvToNixAttrs mdspan

{- | Export CUTLASS package specification.

Called from Nix as: @wasm "cutlass" {}@

Returns an attrset with the full derivation specification.
Header-only NVIDIA CUDA template library.
-}
foreign export ccall "cutlass" cutlassExport :: Value -> IO Value

cutlassExport :: Value -> IO Value
cutlassExport _args = drvToNixAttrs cutlass

{- | Export RapidJSON package specification.

Called from Nix as: @wasm "rapidjson" {}@
-}
foreign export ccall "rapidjson" rapidjsonExport :: Value -> IO Value

rapidjsonExport :: Value -> IO Value
rapidjsonExport _args = drvToNixAttrs rapidjson

{- | Export nlohmann/json package specification.

Called from Nix as: @wasm "nlohmann_json" {}@

Header-only JSON library with pkg-config fixup.
-}
foreign export ccall "nlohmann_json" nlohmannJsonExport :: Value -> IO Value

nlohmannJsonExport :: Value -> IO Value
nlohmannJsonExport _args = drvToNixAttrs nlohmannJson

{- | Export spdlog package specification.

Called from Nix as: @wasm "spdlog" {}@

Super fast C++ logging library, uses external fmt.
-}
foreign export ccall "spdlog" spdlogExport :: Value -> IO Value

spdlogExport :: Value -> IO Value
spdlogExport _args = drvToNixAttrs spdlog

{- | Export Catch2 package specification.

Called from Nix as: @wasm "catch2" {}@

Modern C++ test framework.
-}
foreign export ccall "catch2" catch2Export :: Value -> IO Value

catch2Export :: Value -> IO Value
catch2Export _args = drvToNixAttrs catch2

{- | Export Abseil C++ package specification.

Called from Nix as: @wasm "abseil_cpp" {}@

Google's common C++ libraries.
-}
foreign export ccall "abseil_cpp" abseilCppExport :: Value -> IO Value

abseilCppExport :: Value -> IO Value
abseilCppExport _args = drvToNixAttrs abseilCpp

-- ============================================================================
-- NVIDIA SDK Packages
-- ============================================================================

{- | Export NVIDIA NCCL package specification.

Called from Nix as: @wasm "nvidia_nccl" {}@

Multi-GPU communication library from PyPI wheel.
-}
foreign export ccall "nvidia_nccl" ncclExport :: Value -> IO Value

ncclExport :: Value -> IO Value
ncclExport _args = drvToNixAttrs nccl

{- | Export NVIDIA cuDNN package specification.

Called from Nix as: @wasm "nvidia_cudnn" {}@

Deep learning primitives from PyPI wheel.
-}
foreign export ccall "nvidia_cudnn" cudnnExport :: Value -> IO Value

cudnnExport :: Value -> IO Value
cudnnExport _args = drvToNixAttrs cudnn

{- | Export NVIDIA TensorRT package specification.

Called from Nix as: @wasm "nvidia_tensorrt" {}@

Inference optimization from PyPI wheel.
-}
foreign export ccall "nvidia_tensorrt" tensorrtExport :: Value -> IO Value

tensorrtExport :: Value -> IO Value
tensorrtExport _args = drvToNixAttrs tensorrt

{- | Export NVIDIA cuTensor package specification.

Called from Nix as: @wasm "nvidia_cutensor" {}@

Tensor operations from PyPI wheel.
-}
foreign export ccall "nvidia_cutensor" cutensorExport :: Value -> IO Value

cutensorExport :: Value -> IO Value
cutensorExport _args = drvToNixAttrs cutensor

{- | Export NVIDIA cuSPARSELt package specification.

Called from Nix as: @wasm "nvidia_cusparselt" {}@

Sparse matrix operations from PyPI wheel.
-}
foreign export ccall "nvidia_cusparselt" cusparseltExport :: Value -> IO Value

cusparseltExport :: Value -> IO Value
cusparseltExport _args = drvToNixAttrs cusparselt

{- | Export NVIDIA CUTLASS package specification (typed version).

Called from Nix as: @wasm "nvidia_cutlass" {}@

Header-only CUDA templates library.
This shadows the C++ cutlass package with proper nvidia- naming.
-}
foreign export ccall "nvidia_cutlass" nvidiaCutlassExport :: Value -> IO Value

nvidiaCutlassExport :: Value -> IO Value
nvidiaCutlassExport _args = drvToNixAttrs Nvidia.cutlass

-- ============================================================================
-- Test packages for typed actions
-- ============================================================================

{- | Export jq package specification.

Called from Nix as: @wasm "jq" {}@

Demonstrates typed substitute action.
-}
foreign export ccall "jq" jqExport :: Value -> IO Value

jqExport :: Value -> IO Value
jqExport _args = drvToNixAttrs jq

{- | Export hello-wrapped package specification.

Called from Nix as: @wasm "hello_wrapped" {}@

Demonstrates typed wrap action.
-}
foreign export ccall "hello_wrapped" helloWrappedExport :: Value -> IO Value

helloWrappedExport :: Value -> IO Value
helloWrappedExport _args = drvToNixAttrs helloWrapped
