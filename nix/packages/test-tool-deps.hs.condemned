{-# LANGUAGE OverloadedStrings #-}

{- | Test package demonstrating typed tool dependencies

The Tool action automatically adds the package to nativeBuildInputs.
No need to manually list "jq" in deps - it's inferred from usage.
-}
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "test-tool-deps"
        , version = "1.0.0"
        , specSrc =
            SrcUrl
                UrlSrc
                    { urlUrl = "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"
                    , urlHash = "sha256-jZkUKv2SV28wsM18tCqNxoCZmLxdYH2Idh9RLibH2yA="
                    }
        , phases =
            emptyPhases
                { -- jq is NOT in deps, but Tool action references it
                  -- In zero-bash mode, aleph-exec will resolve the tool
                  install =
                    [ Tool "jq" "jq" [ExprStr "--version"]
                    , Tool "patchelf" "patchelf" [ExprStr "--version"]
                    ]
                }
        , meta =
            Meta
                { description = "Test automatic tool dependency tracking"
                , homepage = Nothing
                , license = "gpl3Plus"
                , maintainers = []
                , platforms = []
                }
        }
