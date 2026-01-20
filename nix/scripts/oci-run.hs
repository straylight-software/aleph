{-# LANGUAGE OverloadedStrings #-}

{- |
Run an OCI container image in a bubblewrap namespace.

Usage: oci-run IMAGE [COMMAND...]

Example: oci-run alpine:latest /bin/sh

Images are cached in ~/.cache/straylight-oci/ for fast subsequent runs.
This is the non-GPU version - see oci-gpu for GPU passthrough.
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Data.List as L
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> script $ do
            echoErr "Usage: oci-run IMAGE [COMMAND...]"
            echoErr ""
            echoErr "Run an OCI container image in a namespace."
            echoErr ""
            echoErr "Examples:"
            echoErr "  oci-run alpine:latest /bin/sh"
            echoErr "  oci-run ubuntu:22.04 bash"
            echoErr "  oci-run nvcr.io/nvidia/pytorch:24.01-py3"
            echoErr ""
            echoErr "Images are cached in ~/.cache/straylight-oci/"
            exit 1
        (image : cmdArgs) -> script $ do
            let cmd = if L.null cmdArgs then ["/bin/bash"] else map pack cmdArgs

            -- Pull or use cached image
            rootfs <- Oci.pullOrCache Oci.defaultConfig (pack image)

            -- Build sandbox
            let sandbox = Oci.baseSandbox rootfs

            -- Execute
            echoErr ":: Entering namespace"
            Bwrap.exec sandbox cmd
