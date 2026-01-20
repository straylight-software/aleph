{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Run an OCI container image in Cloud Hypervisor with GPU passthrough.

Usage: ch-gpu [OPTIONS] [IMAGE] [COMMAND...]

Options:
  --cpus N      Number of vCPUs (default: from config)
  --mem N       Memory in MiB (default: from config)
  --gpu ADDR    GPU PCI address (default: auto-detect)

Environment:
  CONFIG_FILE  - Path to Dhall config (required, set by Nix wrapper)

Example:
  ch-gpu nvcr.io/nvidia/pytorch:24.01-py3
  ch-gpu --gpu 0000:01:00.0 ubuntu:24.04 nvidia-smi

WARNING: This binds the GPU to vfio-pci. The GPU will be unavailable
to the host until the VM exits.
-}
module Main where

import Aleph.Script hiding (FilePath)
import Aleph.Script.Config (StorePath (..), storePathToFilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Vfio as Vfio
import qualified Aleph.Script.Vm as Vm
import Aleph.Script.Vm.Config (CloudHypervisorConfig (..), loadCloudHypervisorConfig)

import Control.Monad (forM_)
import qualified Data.List as L
import Data.Maybe (fromMaybe)
import Numeric.Natural (Natural)
import System.Environment (getArgs, lookupEnv)
import System.Exit (exitFailure)
import Text.Read (readMaybe)

-- | Parse command line arguments
data CliArgs = CliArgs
    { argCpus :: Maybe Int
    , argMem :: Maybe Int
    , argGpu :: Maybe Text
    , argImage :: String
    , argCmd :: [String]
    }

parseArgs :: [String] -> CliArgs
parseArgs = go (CliArgs Nothing Nothing Nothing "nvcr.io/nvidia/pytorch:25.11-py3" [])
  where
    go acc [] = acc
    go acc ("--cpus" : n : rest) = go acc{argCpus = readMaybe n} rest
    go acc ("--mem" : n : rest) = go acc{argMem = readMaybe n} rest
    go acc ("--gpu" : addr : rest) = go acc{argGpu = Just (pack addr)} rest
    go acc (img : rest)
        | Prelude.take 2 img /= "--" =
            -- First non-flag is image, rest is command
            acc{argImage = img, argCmd = rest}
        | otherwise = go acc rest -- skip unknown flags

main :: IO ()
main = do
    -- Load config from Dhall file (set by Nix wrapper)
    configPath <- lookupEnv "CONFIG_FILE"
    case configPath of
        Nothing -> do
            putStrLn "Error: CONFIG_FILE environment variable not set"
            putStrLn "This binary must be run via the Nix-wrapped version."
            exitFailure
        Just path -> do
            cfg <- loadCloudHypervisorConfig path
            runWithConfig cfg

runWithConfig :: CloudHypervisorConfig -> IO ()
runWithConfig cfg@CloudHypervisorConfig{..} = do
    args <- parseArgs <$> getArgs

    -- Merge CLI args with config defaults (16GB default for GPU workloads)
    let cpus = fromMaybe (fromIntegral chDefaultCpus) (argCpus args)
        mem = fromMaybe (max 16384 (fromIntegral chDefaultMemMib)) (argMem args)
        image = argImage args
        cmd = if Prelude.null (argCmd args) then ["nvidia-smi"] else argCmd args

    script $ do
        -- Find GPU
        gpuAddr <- case argGpu args of
            Just addr -> pure addr
            Nothing -> do
                echoErr ":: Auto-detecting NVIDIA GPU"
                gpus <- Vfio.listNvidiaGpus
                case gpus of
                    [] -> die "No NVIDIA GPUs found. Set --gpu 0000:XX:XX.X"
                    (g : _) -> do
                        echoErr $ ":: Found " <> Vfio.pciAddr g
                        pure (Vfio.pciAddr g)

        echoErr $ ":: Cloud Hypervisor + GPU (" <> pack (show cpus) <> " CPUs, " <> pack (show mem) <> " MiB)"
        echoErr $ ":: GPU: " <> gpuAddr

        -- Bind GPU to vfio-pci
        echoErr ":: Binding GPU to vfio-pci"
        boundDevices <- Vfio.bindToVfio gpuAddr

        -- Cleanup: unbind on exit
        let cleanup = do
                echoErr ":: Unbinding GPU from vfio-pci"
                Vfio.unbindFromVfio gpuAddr

        -- Run VM with cleanup
        finally (runVm cfg image gpuAddr cpus mem cmd) cleanup

runVm :: CloudHypervisorConfig -> String -> Vfio.PciAddr -> Int -> Int -> [String] -> Sh ()
runVm CloudHypervisorConfig{..} image gpuAddr cpus mem cmd = do
    withTmpDir $ \workDir -> do
        let rootfsDir = workDir </> "rootfs"
            disk = workDir </> "disk.raw"
            kernelPath = storePathToFilePath chKernel
            busyboxPath = storePathToFilePath chBusybox
            -- Use GPU init script if available, otherwise regular init
            initPath = case chGpuInitScript of
                Just p -> storePathToFilePath p
                Nothing -> storePathToFilePath chInitScript

        -- Pull image
        echoErr $ ":: Pulling " <> pack image
        mkdirP rootfsDir
        setEnv "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt"
        bash_ $ "crane export --platform linux/amd64 '" <> pack image <> "' - | tar -xf - -C " <> pack rootfsDir

        -- Inject busybox and init
        echoErr ":: Injecting init"
        Vm.injectBusybox busyboxPath rootfsDir
        cp initPath (rootfsDir </> "init")
        run_ "chmod" ["+x", pack (rootfsDir </> "init")]

        -- Write build command
        let buildCmdPath = rootfsDir </> "build-cmd"
            cmdScript =
                "#!/bin/bash\nset -euo pipefail\ncd /workspace 2>/dev/null || cd /root\n"
                    <> Prelude.unwords cmd
                    <> "\n"
        liftIO $ writeFile buildCmdPath cmdScript
        run_ "chmod" ["+x", pack buildCmdPath]

        -- Build disk image (16GB for GPU workloads)
        echoErr ":: Building rootfs"
        Vm.buildExt4Sized (16 * 1024 * 1024) rootfsDir disk

        -- Boot with GPU
        echoErr ":: Booting Cloud Hypervisor with GPU passthrough"
        let vmCfg =
                Vm.defaultCloudHypervisorConfig
                    { Vm.chKernel = kernelPath
                    , Vm.chDisk = disk
                    , Vm.chCpus = cpus
                    , Vm.chMemMib = mem
                    }
        Vm.runCloudHypervisorGpu vmCfg gpuAddr
