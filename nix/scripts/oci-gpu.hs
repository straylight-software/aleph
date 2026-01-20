{-# LANGUAGE OverloadedStrings #-}

{- |
Run an OCI container image with NVIDIA GPU access.

Usage: oci-gpu IMAGE [COMMAND...]

Example: oci-gpu nvcr.io/nvidia/pytorch:24.01-py3 nvidia-smi

This script:
1. Pulls and caches the OCI image
2. Discovers NVIDIA GPU devices and drivers
3. Runs the container in a bubblewrap namespace with GPU passthrough
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import Data.Function ((&))
import qualified Data.List as L
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> script $ do
            echoErr "Usage: oci-gpu IMAGE [COMMAND...]"
            echoErr ""
            echoErr "Run an OCI container image with NVIDIA GPU access."
            echoErr ""
            echoErr "Examples:"
            echoErr "  oci-gpu nvcr.io/nvidia/pytorch:24.01-py3"
            echoErr "  oci-gpu nvcr.io/nvidia/cuda:12.0-base nvidia-smi"
            echoErr "  oci-gpu alpine:latest /bin/sh"
            echoErr ""
            echoErr "Images are cached in ~/.cache/straylight-oci/"
            exit 1
        (image : cmdArgs) -> script $ do
            let cmd = if L.null cmdArgs then ["nvidia-smi"] else map pack cmdArgs
                imageText = pack image

            -- Pull or use cached image
            rootfs <- Oci.pullOrCache Oci.defaultConfig imageText

            -- Create nvidia mount points
            mkdirP (rootfs </> "usr/local/nvidia/bin")
            mkdirP (rootfs </> "usr/local/nvidia/lib64")

            -- Discover GPU binds and container environment
            gpuBinds <- withGpuBinds
            containerEnv <- Oci.getContainerEnv imageText

            -- Build sandbox with GPU support
            let sandbox =
                    Oci.baseSandbox rootfs
                        & Oci.withGpuSupport containerEnv gpuBinds

            -- Execute
            echoErr ":: Entering namespace with GPU"
            Bwrap.exec sandbox cmd
