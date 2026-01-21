{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "nvidia-cutlass"
        , version = "4.3.3"
        , specSrc =
            SrcUrl
                UrlSrc
                    { urlUrl = "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.3.3.zip"
                    , urlHash = "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w="
                    }
        , deps =
            [ buildDep "unzip"
            ]
        , phases =
            emptyPhases
                { unpack = [] -- Don't auto-unpack
                , install =
                    [ Unzip src (RefRel "unpacked")
                    , Mkdir (outSub "include") True
                    , Copy (RefRel "unpacked/cutlass-4.3.3/include/cutlass") (outSub "include/cutlass")
                    , Copy (RefRel "unpacked/cutlass-4.3.3/include/cute") (outSub "include/cute")
                    ]
                }
        , meta =
            Meta
                { description = "NVIDIA CUTLASS 4.3.3 - CUDA linear algebra templates"
                , homepage = Just "https://github.com/NVIDIA/cutlass"
                , license = "bsd3"
                , maintainers = []
                , platforms = []
                }
        }
