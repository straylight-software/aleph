{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | hello-wrapped - Example package demonstrating wrapProgram

Demonstrates:
- Simple autotools build
- postFixup with typed wrap action
- Setting environment variables in the wrapper

This is a test package for proving the typed wrap action works.
-}
module Aleph.Nix.Packages.HelloWrapped (helloWrapped) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax

helloWrapped :: Drv
helloWrapped =
    mkDerivation
        [ pname "hello-wrapped"
        , version "2.12.1"
        , src $
            fetchurl
                [ url "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"
                , hash "sha256-jZkUKv2SV28wsM18tCqNxoCZmLxdYH2Idh9RLibH2yA="
                ]
        , nativeBuildInputs ["makeWrapper"]
        , -- Typed wrap action - no string escaping bugs!
          postFixup
            [ wrap
                "bin/hello"
                [ wrapPrefix "PATH" "/some/extra/path"
                , wrapSet "HELLO_WRAPPED" "true"
                , wrapSetDefault "LANG" "en_US.UTF-8"
                ]
            ]
        , description "GNU Hello with wrapper demonstration"
        , homepage "https://www.gnu.org/software/hello/"
        , license "gpl3Plus"
        , mainProgram "hello"
        ]
