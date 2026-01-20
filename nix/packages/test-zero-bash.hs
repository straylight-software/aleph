{-# LANGUAGE OverloadedStrings #-}

{- | Test package for zero-bash architecture

This package tests the aleph-exec builder with typed actions.
No stdenv hooks, no bash - pure Haskell execution.

Usage: test-zero-bash = call-package ./test-zero-bash.hs { zeroBash = true; };
-}
module Pkg where

import Prelude hiding (writeFile)
import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "test-zero-bash"
        , version "1.0.0"
        -- No source - we create everything in installPhase
        , dontUnpack True
        , installPhase
            [ -- Create directory structure
              mkdir "bin"
            , mkdir "lib"
            , mkdir "include"
            , mkdir "share/doc/test-zero-bash"
            -- Create a header file
            , writeFile "include/test.h" "#pragma once\n#define TEST_VERSION \"1.0.0\"\n"
            -- Create a fake library
            , writeFile "lib/libtest.so.1.0.0" "PLACEHOLDER"
            -- Create symlinks
            , symlink "libtest.so.1.0.0" "lib/libtest.so.1"
            , symlink "libtest.so.1" "lib/libtest.so"
            -- Create documentation
            , writeFile "share/doc/test-zero-bash/README" "Test package for zero-bash architecture.\n"
            ]
        , postInstall
            [ -- Test substitution
              substitute "include/test.h"
                [ ("TEST_VERSION", "ZERO_BASH_VERSION")
                ]
            ]
        , description "Test package for zero-bash architecture"
        , license "mit"
        ]
