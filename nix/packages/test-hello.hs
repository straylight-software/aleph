{-# LANGUAGE OverloadedStrings #-}

{- | GNU Hello - test package for call-package

Usage: hello = call-package ./test-hello.hs {};
-}
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "test-hello"
        , version "1.0.0"
        , src $
            fetchurl
                [ url "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"
                , hash "sha256-jZkUKv2SV28wsM18tCqNxoCZmLxdYH2Idh9RLibH2yA="
                ]
        , description "Test package for call-package"
        , license "gpl3Plus"
        ]
