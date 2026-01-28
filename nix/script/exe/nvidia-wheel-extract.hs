{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
nvidia-wheel-extract - Extract NVIDIA libraries from PyPI wheels

This script downloads and extracts NVIDIA libraries from pypi.nvidia.com
wheels, which have no redistribution restrictions.

Usage:
  nvidia-wheel-extract <package> <output-dir>
  nvidia-wheel-extract all <output-dir>

Packages:
  nccl       - NCCL 2.28.9
  cudnn      - cuDNN 9.17.0.29
  tensorrt   - TensorRT 10.14.1.48
  cutensor   - cuTensor 2.4.1
  cusparselt - cuSPARSELt 0.8.1
  all        - Extract all packages

Examples:
  nvidia-wheel-extract nccl ./nccl-sdk
  nvidia-wheel-extract all ./nvidia-libs
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Nvidia.Versions as V
import qualified Aleph.Script.Nvidia.Wheel as Wheel
import qualified Data.Text as T
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [pkg, outDir] -> script $ verbosely $ extractPackage pkg (fromText $ pack outDir)
        _ -> printHelp

printHelp :: IO ()
printHelp = do
    putStrLn "nvidia-wheel-extract - Extract NVIDIA libraries from PyPI wheels"
    putStrLn ""
    putStrLn "Usage:"
    putStrLn "  nvidia-wheel-extract <package> <output-dir>"
    putStrLn ""
    putStrLn "Packages:"
    putStrLn $ "  nccl       - NCCL " ++ T.unpack V.ncclVersion
    putStrLn $ "  cudnn      - cuDNN " ++ T.unpack V.cudnnVersion
    putStrLn $ "  tensorrt   - TensorRT " ++ T.unpack V.tensorrtVersion
    putStrLn $ "  cutensor   - cuTensor " ++ T.unpack V.cutensorVersion
    putStrLn $ "  cusparselt - cuSPARSELt " ++ T.unpack V.cusparseltVersion
    putStrLn "  all        - Extract all packages"
    putStrLn ""
    putStrLn "Examples:"
    putStrLn "  nvidia-wheel-extract nccl ./nccl-sdk"
    putStrLn "  nvidia-wheel-extract all ./nvidia-libs"

extractPackage :: String -> FilePath -> Sh ()
extractPackage pkg outDir = case pkg of
    "nccl" -> extractOne Wheel.nccl outDir
    "cudnn" -> extractOne Wheel.cudnn outDir
    "tensorrt" -> extractOne Wheel.tensorrt outDir
    "cutensor" -> extractOne Wheel.cutensor outDir
    "cusparselt" -> extractOne Wheel.cusparselt outDir
    "all" -> Wheel.extractAllWheels V.currentPlatform outDir
    _ -> do
        echoErr $ "Unknown package: " <> pack pkg
        echoErr "Use one of: nccl, cudnn, tensorrt, cutensor, cusparselt, all"
        exit 1

extractOne :: Wheel.WheelDef -> FilePath -> Sh ()
extractOne def outDir =
    case Wheel.forPlatform V.currentPlatform def of
        Just spec -> do
            mkdirP outDir
            Wheel.extractToNixLayout spec outDir
            echoErr $ ":: Done! Extracted to " <> toTextIgnore outDir
        Nothing -> do
            echoErr $ ":: Error: " <> Wheel.defName def <> " not available for this platform"
            exit 1
