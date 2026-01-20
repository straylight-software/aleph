{-# LANGUAGE OverloadedStrings #-}

{- | Test package demonstrating typed tool dependencies

The 'tool' function automatically adds the package to nativeBuildInputs.
No need to manually list "jq" in deps - it's inferred from usage.
-}
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "test-tool-deps"
        , version "1.0.0"
        , src $
            fetchurl
                [ url "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"
                , hash "sha256-jZkUKv2SV28wsM18tCqNxoCZmLxdYH2Idh9RLibH2yA="
                ]
        , -- jq is NOT in nativeBuildInputs, but will be added automatically
          -- because we use 'tool "jq"' in postInstall
          postInstall
            [ tool "jq" ["--version"] -- jq added to nativeBuildInputs automatically
            , tool "patchelf" ["--version"] -- patchelf too
            ]
        , description "Test automatic tool dependency tracking"
        , license "gpl3Plus"
        ]
