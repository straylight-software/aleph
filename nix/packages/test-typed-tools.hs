{-# LANGUAGE OverloadedStrings #-}

{- | Test package demonstrating typed tool APIs.

This package tests that:
1. Typed tool modules compile correctly
2. Tool dependencies are automatically tracked
3. Actions serialize correctly to Nix

Expected nativeBuildInputs: jq, patchelf (automatically from typed tools)
-}
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "test-typed-tools"
    , version = "1.0.0"
    , specSrc = SrcNone  -- No source needed
    , phases = emptyPhases
        { unpack = []  -- No source to unpack
        , install = 
            [ -- Create directory structure
              Mkdir (outSub "share/test-typed-tools") True
            , -- Create a test JSON file
              Write (outSub "share/test-typed-tools/metadata.json") 
                    "{\"version\": \"1.0.0\", \"name\": \"test\"}"
            ]
        , fixup = 
            [ -- Test that tools are available
              Tool "jq" "jq" [ExprStr "--version"]
              -- patchelf would be tested too if we had ELF binaries
            ]
        }
    , meta = Meta
        { description = "Test package for typed tool APIs"
        , homepage = Nothing
        , license = "mit"
        , maintainers = []
        , platforms = []
        }
    }
