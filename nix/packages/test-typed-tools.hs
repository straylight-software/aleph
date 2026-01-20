{-# LANGUAGE OverloadedStrings #-}

{- | Test package demonstrating typed tool APIs.

This package tests that:
1. Typed tool modules compile correctly
2. Tool dependencies are automatically tracked
3. Actions serialize correctly to Nix

Expected nativeBuildInputs: jq, patchelf (automatically from typed tools)
-}
module Pkg where

import Aleph.Nix.Package
import qualified Aleph.Nix.Tools.Install as Install
import qualified Aleph.Nix.Tools.Jq as Jq
import qualified Aleph.Nix.Tools.PatchElf as PatchElf
import qualified Aleph.Nix.Tools.Substitute as Sub

pkg :: Drv
pkg =
    mkDerivation
        [ pname "test-typed-tools"
        , version "1.0.0"
        , -- No source - this is a meta-test package
          dontUnpack True
        , -- Test installPhase with Install helpers
          installPhase
            [ Install.dir "share/test-typed-tools"
            , -- Create a test JSON file
              run "sh" ["-c", "echo '{\"version\": \"1.0.0\", \"name\": \"test\"}' > $out/share/test-typed-tools/metadata.json"]
            ]
        , -- Test postInstall with jq (reads the JSON we created)
          -- This verifies jq was added to nativeBuildInputs
          postInstall
            [ Jq.query Jq.defaults{Jq.rawOutput = True} ".version" "$out/share/test-typed-tools/metadata.json"
            ]
        , -- Test postFixup with PatchElf - just verify the tool is available
          -- (patchelf --version exits 0 and prints version)
          postFixup
            [ run "patchelf" ["--version"]
            ]
        , description "Test package for typed tool APIs"
        , license "mit"
        ]
