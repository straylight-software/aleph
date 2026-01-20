{-# LANGUAGE OverloadedStrings #-}

{- | GNU Hello - test package for call-package

Usage: hello = call-package ./test-hello.hs {};
-}
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "test-hello"
    , version = "1.0.0"
    , specSrc = SrcUrl UrlSrc
        { urlUrl = "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"
        , urlHash = "sha256-jZkUKv2SV28wsM18tCqNxoCZmLxdYH2Idh9RLibH2yA="
        }
    -- Uses default autotools phases (configure/make/make install)
    , meta = Meta
        { description = "Test package for call-package"
        , homepage = Nothing
        , license = "gpl3Plus"
        , maintainers = []
        , platforms = []
        }
    }
