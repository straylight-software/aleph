{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Run an OCI container image in a Cloud Hypervisor VM.

Usage: ch-run [OPTIONS] [IMAGE]

Options:
  --cpus N     Number of vCPUs (default: from config)
  --mem N      Memory in MiB (default: from config)

Environment:
  CONFIG_FILE  - Path to Dhall config (required, set by Nix wrapper)

Example: ch-run ubuntu:24.04
         ch-run --cpus 4 --mem 8192 alpine:latest

The CONFIG_FILE must be a Dhall expression of type:
  { chKernel : Text
  , chBusybox : Text
  , chInitScript : Text
  , chGpuInitScript : Optional Text
  , chDefaultCpus : Natural
  , chDefaultMemMib : Natural
  , chHugepages : Bool
  , chCacheDir : Text
  }
-}
module Main where

import Aleph.Script hiding (FilePath)
import Aleph.Script.Config (StorePath (..), storePathToFilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Vm as Vm
import Aleph.Script.Vm.Config (CloudHypervisorConfig (..), loadCloudHypervisorConfig)

import Data.Maybe (fromMaybe)
import Numeric.Natural (Natural)
import System.Environment (getArgs, lookupEnv)
import System.Exit (exitFailure)
import Text.Read (readMaybe)

-- | Parse command line arguments
data CliArgs = CliArgs
    { argCpus :: Maybe Int
    , argMem :: Maybe Int
    , argImage :: String
    }

parseArgs :: [String] -> CliArgs
parseArgs = go (CliArgs Nothing Nothing "ubuntu:24.04")
  where
    go acc [] = acc
    go acc ("--cpus" : n : rest) = go acc{argCpus = readMaybe n} rest
    go acc ("--mem" : n : rest) = go acc{argMem = readMaybe n} rest
    go acc (img : rest)
        | Prelude.take 2 img /= "--" = go acc{argImage = img} rest
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

    -- Merge CLI args with config defaults
    let cpus = fromMaybe (fromIntegral chDefaultCpus) (argCpus args)
        mem = fromMaybe (fromIntegral chDefaultMemMib) (argMem args)
        image = argImage args

    script $ do
        echoErr $ ":: Cloud Hypervisor VM (" <> pack (show cpus) <> " CPUs, " <> pack (show mem) <> " MiB)"

        withTmpDir $ \workDir -> do
            let rootfsDir = workDir </> "rootfs"
                disk = workDir </> "disk.raw"
                kernelPath = storePathToFilePath chKernel
                busyboxPath = storePathToFilePath chBusybox
                initPath = storePathToFilePath chInitScript

            -- Pull image
            echoErr $ ":: Pulling " <> pack image
            mkdirP rootfsDir
            setEnv "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt"
            Oci.pullOrCache Oci.defaultConfig (pack image)

            -- Export to rootfs
            bash_ $ "crane export --platform linux/amd64 '" <> pack image <> "' - | tar -xf - -C " <> pack rootfsDir

            -- Inject busybox and init
            echoErr ":: Injecting init"
            Vm.injectBusybox busyboxPath rootfsDir
            cp initPath (rootfsDir </> "init")
            run_ "chmod" ["+x", pack (rootfsDir </> "init")]

            -- Build disk image (8GB for Cloud Hypervisor)
            echoErr ":: Building rootfs"
            Vm.buildExt4Sized (8 * 1024 * 1024) rootfsDir disk

            -- Boot
            echoErr ":: Booting Cloud Hypervisor"
            let vmCfg =
                    Vm.defaultCloudHypervisorConfig
                        { Vm.chKernel = kernelPath
                        , Vm.chDisk = disk
                        , Vm.chCpus = cpus
                        , Vm.chMemMib = mem
                        }
            Vm.runCloudHypervisor vmCfg
