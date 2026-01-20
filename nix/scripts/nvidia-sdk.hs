{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
nvidia-sdk - Unified NVIDIA SDK extraction

This script provides unified extraction of NVIDIA libraries from:
- PyPI wheels (preferred, no redistribution issues)
- NGC containers (for CUDA toolkit with nvcc)

Usage:
  nvidia-sdk wheels <output-dir>           Extract all wheels (nccl, cudnn, tensorrt, etc)
  nvidia-sdk wheel <name> <output-dir>     Extract single wheel
  nvidia-sdk container <image> <output>    Extract from container
  nvidia-sdk triton <rootfs> <output>      Extract tritonserver from rootfs
  nvidia-sdk cuda <rootfs> <output>        Extract CUDA toolkit from rootfs
  nvidia-sdk info                          Show version information

Examples:
  nvidia-sdk wheels ./nvidia-libs
  nvidia-sdk wheel nccl ./nccl
  nvidia-sdk container nvcr.io/nvidia/tritonserver:25.11-py3 ./triton
  nvidia-sdk info
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Nvidia.Container as Container
import qualified Aleph.Script.Nvidia.Versions as V
import qualified Aleph.Script.Nvidia.Wheel as Wheel
import qualified Data.Text as T
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["wheels", outDir] ->
            script $ verbosely $ Wheel.extractAllWheels V.currentPlatform (fromText $ pack outDir)
        ["wheel", name, outDir] ->
            script $ verbosely $ extractWheel (pack name) (fromText $ pack outDir)
        ["container", imageRef, outDir] ->
            script $ verbosely $ extractContainer (pack imageRef) (fromText $ pack outDir)
        ["triton", rootfs, outDir] ->
            script $ verbosely $ Container.extractSdk (fromText $ pack rootfs) (fromText $ pack outDir) Container.Tritonserver
        ["cuda", rootfs, outDir] ->
            script $ verbosely $ Container.extractSdk (fromText $ pack rootfs) (fromText $ pack outDir) Container.CudaToolkit
        ["runtime", rootfs, outDir] ->
            script $ verbosely $ Container.extractSdk (fromText $ pack rootfs) (fromText $ pack outDir) Container.CudaRuntime
        ["info"] -> printInfo
        _ -> printHelp

printHelp :: IO ()
printHelp = do
    putStrLn "nvidia-sdk - Unified NVIDIA SDK extraction"
    putStrLn ""
    putStrLn "Usage:"
    putStrLn "  nvidia-sdk wheels <output-dir>           Extract all wheels"
    putStrLn "  nvidia-sdk wheel <name> <output-dir>     Extract single wheel"
    putStrLn "  nvidia-sdk container <image> <output>    Pull and extract from container"
    putStrLn "  nvidia-sdk triton <rootfs> <output>      Extract tritonserver from rootfs"
    putStrLn "  nvidia-sdk cuda <rootfs> <output>        Extract CUDA toolkit from rootfs"
    putStrLn "  nvidia-sdk runtime <rootfs> <output>     Extract CUDA runtime from rootfs"
    putStrLn "  nvidia-sdk info                          Show version information"
    putStrLn ""
    putStrLn "Available wheels:"
    putStrLn "  nccl, cudnn, tensorrt, cutensor, cusparselt"
    putStrLn ""
    putStrLn "Examples:"
    putStrLn "  nvidia-sdk wheels ./nvidia-libs"
    putStrLn "  nvidia-sdk wheel nccl ./nccl"
    putStrLn "  nvidia-sdk container nvcr.io/nvidia/tritonserver:25.11-py3 ./triton"

printInfo :: IO ()
printInfo = do
    putStrLn "NVIDIA SDK Versions"
    putStrLn "==================="
    putStrLn ""
    putStrLn "Target Versions (NGC 25.11 - Blackwell blessed):"
    putStrLn $ "  CUDA:       " ++ T.unpack V.cudaVersion
    putStrLn $ "  cuDNN:      " ++ T.unpack V.cudnnVersion
    putStrLn $ "  NCCL:       " ++ T.unpack V.ncclVersion
    putStrLn $ "  TensorRT:   " ++ T.unpack V.tensorrtVersion
    putStrLn $ "  cuTensor:   " ++ T.unpack V.cutensorVersion
    putStrLn $ "  cuSPARSELt: " ++ T.unpack V.cusparseltVersion
    putStrLn $ "  CUTLASS:    " ++ T.unpack V.cutlassVersion
    putStrLn ""
    putStrLn "Container References:"
    let triton = V.tritonserver V.currentPlatform
    putStrLn $ "  tritonserver: " ++ T.unpack (V.containerRef triton)
    let cuda = V.cudaDevel V.currentPlatform
    putStrLn $ "  cuda-devel:   " ++ T.unpack (V.containerRef cuda)
    putStrLn ""
    putStrLn "SM Architectures:"
    putStrLn $ "  Volta:     " ++ T.unpack (V.smArchName V.volta)
    putStrLn $ "  Turing:    " ++ T.unpack (V.smArchName V.turing)
    putStrLn $ "  Ampere:    " ++ T.unpack (V.smArchName V.ampere)
    putStrLn $ "  Ada:       " ++ T.unpack (V.smArchName V.ada)
    putStrLn $ "  Hopper:    " ++ T.unpack (V.smArchName V.hopper)
    putStrLn $ "  Blackwell: " ++ T.unpack (V.smArchName V.blackwell)

extractWheel :: Text -> FilePath -> Sh ()
extractWheel name outDir = do
    let wheelDef = case name of
            "nccl" -> Just Wheel.nccl
            "cudnn" -> Just Wheel.cudnn
            "tensorrt" -> Just Wheel.tensorrt
            "cutensor" -> Just Wheel.cutensor
            "cusparselt" -> Just Wheel.cusparselt
            _ -> Nothing

    case wheelDef of
        Nothing -> do
            echoErr $ "Unknown wheel: " <> name
            echoErr "Available: nccl, cudnn, tensorrt, cutensor, cusparselt"
            exit 1
        Just def ->
            case Wheel.forPlatform V.currentPlatform def of
                Nothing -> do
                    echoErr $ "Wheel " <> name <> " not available for this platform"
                    exit 1
                Just spec -> do
                    mkdirP outDir
                    Wheel.extractToNixLayout spec outDir
                    echoErr $ ":: Done! Extracted to " <> toTextIgnore outDir

extractContainer :: Text -> FilePath -> Sh ()
extractContainer imageRef outDir = do
    echoErr $ ":: Pulling and extracting " <> imageRef

    withTmpDir $ \tmpDir -> do
        let rootfs = tmpDir </> "rootfs"

        -- Pull container
        Container.pullImage imageRef rootfs

        -- Determine extraction mode from image name
        let mode
                | T.isInfixOf "triton" imageRef = Container.Tritonserver
                | T.isInfixOf "devel" imageRef = Container.CudaToolkit
                | otherwise = Container.CudaRuntime

        -- Extract SDK
        Container.extractSdk rootfs outDir mode

    echoErr $ ":: Done! Extracted to " <> toTextIgnore outDir
