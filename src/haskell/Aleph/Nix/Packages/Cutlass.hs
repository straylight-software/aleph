{-# LANGUAGE OverloadedStrings #-}

{- | CUTLASS package definition - NVIDIA CUDA Templates for Linear Algebra.

CUTLASS provides highly optimized CUDA template primitives for
matrix multiply and convolution operations. Header-only library.

= Usage from Nix

@
let
  wasm = builtins.wasm ./packages.wasm;
  cutlassSpec = wasm "cutlass" {};
in
  straylight-lib.packages.build cutlassSpec
@

= CA Soundness

This package definition is sound under content-addressed derivations:

- Source hash is explicit (sha256Hash)
- No implicit dependencies
- Header-only install is deterministic
-}
module Aleph.Nix.Packages.Cutlass (
    cutlass,
    cutlassVersion,
) where

import Aleph.Nix.Derivation
import Data.Text (Text)

-- | Current version of CUTLASS we package.
cutlassVersion :: Text
cutlassVersion = "4.3.3"

{- | CUTLASS: CUDA Templates for Linear Algebra Subroutines.

Features:
  - High-performance GEMM (matrix multiply) templates
  - Convolution operations
  - Tensor Core support
  - CuTe tensor abstraction layer

Note: Header-only library, no build required.
-}
cutlass :: Drv
cutlass =
    defaultDrv
        { drvName = "cutlass"
        , drvVersion = cutlassVersion
        , drvSrc =
            SrcGitHub
                FetchGitHub
                    { ghOwner = "NVIDIA"
                    , ghRepo = "cutlass"
                    , ghRev = "v" <> cutlassVersion
                    , ghHash = sha256Hash "uOfSEjbwn/edHEgBikC9wAarn6c6T71ebPg74rv2qlw="
                    }
        , drvBuilder = NoBuilder -- header-only
        , drvDeps = emptyDeps
        , drvMeta =
            Meta
                { description = "CUDA Templates for Linear Algebra Subroutines"
                , homepage = Just "https://github.com/NVIDIA/cutlass"
                , license = "bsd3"
                , platforms = [] -- all platforms (headers)
                , mainProgram = Nothing
                }
        , drvPhases =
            emptyPhases
                { postInstall =
                    [ Mkdir "include"
                    , Copy "include/cutlass" "include/cutlass"
                    , Copy "include/cute" "include/cute"
                    ]
                }
        , drvStrictDeps = True
        , drvDoCheck = False
        , drvSystem = Nothing
        }
