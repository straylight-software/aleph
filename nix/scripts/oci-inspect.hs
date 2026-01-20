{-# LANGUAGE OverloadedStrings #-}

{- |
Inspect an OCI container image (manifest and config).

Usage: oci-inspect IMAGE

Example: oci-inspect alpine:latest
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Tools.Crane as Crane
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [image] -> script $ do
            let imageText = pack image

            -- Set SSL cert
            setEnv "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt"

            echo "═══════════════════════════════════════════════════════════"
            echo $ " OCI Image: " <> imageText
            echo "═══════════════════════════════════════════════════════════"
            echo ""

            -- Get and display manifest
            manifest <- Crane.manifest imageText
            echo manifest

            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo " Config"
            echo "═══════════════════════════════════════════════════════════"
            echo ""

            -- Get and display config
            config <- Crane.config imageText
            echo config
        _ -> script $ do
            echoErr "Usage: oci-inspect IMAGE"
            echoErr ""
            echoErr "Show manifest and config for an OCI image."
            echoErr ""
            echoErr "Example: oci-inspect alpine:latest"
            exit 1
