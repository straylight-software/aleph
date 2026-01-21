{-# LANGUAGE OverloadedStrings #-}

{- | Test package for zero-bash architecture

This package tests the aleph-exec builder with typed actions.
No stdenv hooks, no bash - pure Haskell execution.

Usage: test-zero-bash = call-package ./test-zero-bash.hs { zeroBash = true; };
-}
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "test-zero-bash"
        , version = "1.0.0"
        , specSrc = SrcNone -- No source - we create everything in installPhase
        , phases =
            emptyPhases
                { unpack = [] -- No source to unpack
                , install =
                    [ -- Create directory structure
                      Mkdir (outSub "bin") True
                    , Mkdir (outSub "lib") True
                    , Mkdir (outSub "include") True
                    , Mkdir (outSub "share/doc/test-zero-bash") True
                    , -- Create a header file
                      Write
                        (outSub "include/test.h")
                        "#pragma once\n#define TEST_VERSION \"1.0.0\"\n"
                    , -- Create a fake library placeholder
                      Write (outSub "lib/libtest.so.1.0.0") "PLACEHOLDER"
                    , -- Create symlinks
                      Symlink (RefLit "libtest.so.1.0.0") (outSub "lib/libtest.so.1")
                    , Symlink (RefLit "libtest.so.1") (outSub "lib/libtest.so")
                    , -- Create documentation
                      Write
                        (outSub "share/doc/test-zero-bash/README")
                        "Test package for zero-bash architecture.\n"
                    ]
                , fixup =
                    [ -- Test substitution
                      Substitute
                        (outSub "include/test.h")
                        [ ("TEST_VERSION", "ZERO_BASH_VERSION")
                        ]
                    ]
                }
        , meta =
            Meta
                { description = "Test package for zero-bash architecture"
                , homepage = Nothing
                , license = "mit"
                , maintainers = []
                , platforms = []
                }
        }
