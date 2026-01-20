{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | jq - Lightweight and flexible command-line JSON processor

Demonstrates:
- Autotools builder
- postPatch with typed substitute
- postInstall with typed actions

Original Nix uses:
- configureFlags
- postPatch with substituteInPlace
- Standard autotools phases
-}
module Aleph.Nix.Packages.Jq (jq) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax

jq :: Drv
jq =
    mkDerivation
        [ pname "jq"
        , version "1.7.1"
        , src $
            fetchFromGitHub
                [ owner "jqlang"
                , repo "jq"
                , rev "jq-1.7.1"
                , hash "sha256-/lYdn+IWs/+05yDCetWxQjGhqOGyjJwmKvLpBPzwHlk="
                ]
        , nativeBuildInputs ["autoreconfHook"]
        , buildInputs ["oniguruma"]
        , -- Autotools with configure flags
          configureFlags
            [ "--with-oniguruma"
            , "--disable-docs" -- Skip building docs
            ]
        , -- Typed substituteInPlace - no shell escaping bugs!
          postPatch
            [ substitute
                "Makefile.am"
                [ ("bin_PROGRAMS = jq", "bin_PROGRAMS = jq") -- no-op example
                ]
            ]
        , -- Typed postInstall actions
          postInstall
            [ mkdir "share/doc/jq"
            , copy "README.md" "share/doc/jq/"
            ]
        , description "Lightweight and flexible command-line JSON processor"
        , homepage "https://jqlang.github.io/jq/"
        , license "mit"
        , mainProgram "jq"
        ]
