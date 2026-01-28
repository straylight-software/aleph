{-# LANGUAGE OverloadedStrings #-}

{- |
Pull and extract an OCI container image.

Usage: oci-pull IMAGE [OUTPUT_DIR]

Example: oci-pull alpine:latest ./rootfs
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Tools.Crane as Crane
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> script $ usage
        [image] -> script $ pullImage image "."
        [image, output] -> script $ pullImage image output
        _ -> script $ usage
  where
    usage = do
        echoErr "Usage: oci-pull IMAGE [OUTPUT_DIR]"
        echoErr ""
        echoErr "Pull and extract an OCI image to a directory."
        echoErr ""
        echoErr "Examples:"
        echoErr "  oci-pull alpine:latest              # extract to current dir"
        echoErr "  oci-pull alpine:latest ./rootfs     # extract to ./rootfs"
        exit 1

    pullImage image output = do
        let imageText = pack image

        -- Set SSL cert
        setEnv "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt"

        echoErr $ ":: Pulling " <> imageText
        Crane.exportToDir Crane.defaults imageText output
        echoErr $ ":: Extracted to " <> pack output
