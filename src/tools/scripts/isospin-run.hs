{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Run an OCI container image in a Firecracker microVM.

Usage: fc-run [OPTIONS] [IMAGE]
       fc-run --build IMAGE COMMAND...

Options:
  --cpus N     Number of vCPUs (default: from config or 2)
  --mem N      Memory in MiB (default: from config or 1024)
  --build      Build mode: run COMMAND and exit

Environment:
  CONFIG_FILE  - Path to Dhall config (required, set by Nix wrapper)

Example: fc-run ubuntu:24.04
         fc-run --cpus 4 --mem 2048 alpine:latest
         fc-run --build ubuntu:24.04 make -j8

The CONFIG_FILE must be a Dhall expression of type:
  { kernel : Text
  , busybox : Text
  , initScript : Text
  , buildInitScript : Text
  , defaultCpus : Natural
  , defaultMemMib : Natural
  , cacheDir : Text
  }
-}
module Main where

import Aleph.Script hiding (FilePath)
import Aleph.Script.Config (StorePath (..), storePathToFilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Vm as Vm
import Aleph.Script.Vm.Config (FirecrackerConfig (..), loadFirecrackerConfig)
import qualified Data.Text as T

import Control.Exception (bracket_)
import Data.Maybe (fromMaybe)
import Numeric.Natural (Natural)
import System.Environment (getArgs, lookupEnv)
import System.Exit (exitFailure)
import Text.Read (readMaybe)

-- | Parse command line arguments
data CliArgs = CliArgs
    { argCpus :: Maybe Int
    , argMem :: Maybe Int
    , argBuild :: Bool
    , argNoNet :: Bool
    , argImage :: String
    , argCommand :: [String]
    }

parseArgs :: [String] -> CliArgs
parseArgs = go (CliArgs Nothing Nothing False False "ubuntu:24.04" [])
  where
    go acc [] = acc
    go acc ("--cpus" : n : rest) = go acc{argCpus = readMaybe n} rest
    go acc ("--mem" : n : rest) = go acc{argMem = readMaybe n} rest
    go acc ("--build" : rest) = go acc{argBuild = True} rest
    go acc ("--no-net" : rest) = go acc{argNoNet = True} rest
    go acc (img : rest)
        | Prelude.take 2 img /= "--" =
            if argBuild acc
                then -- In build mode, first non-flag is image, rest is command
                    acc{argImage = img, argCommand = rest}
                else go acc{argImage = img} rest
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
            cfg <- loadFirecrackerConfig path
            runWithConfig cfg

runWithConfig :: FirecrackerConfig -> IO ()
runWithConfig FirecrackerConfig{..} = do
    args <- parseArgs <$> getArgs

    -- Merge CLI args with config defaults
    let cpus = fromMaybe (fromIntegral defaultCpus) (argCpus args)
        mem = fromMaybe (fromIntegral defaultMemMib) (argMem args)
        image = argImage args
        isBuild = argBuild args
        buildCmd = argCommand args
        enableNet = not (argNoNet args)

    when (isBuild && Prelude.null buildCmd) $ do
        putStrLn "Error: --build requires a command"
        putStrLn "Usage: fc-run --build IMAGE COMMAND..."
        exitFailure

    script $ do
        let modeStr = if isBuild then "build" else "run"
        echoErr $ ":: Firecracker " <> modeStr <> " (" <> pack (show cpus) <> " CPUs, " <> pack (show mem) <> " MiB)"

        -- Network config (use default subnet 172.16.0.0/30)
        let netCfg = Vm.defaultFirecrackerNetwork

        withTmpDir $ \workDir -> do
            let rootfsDir = workDir </> "rootfs"
                disk = workDir </> "disk.ext4"
                kernelPath = storePathToFilePath kernel
                busyboxPath = storePathToFilePath busybox
                -- Use different init script for build vs run
                initPath =
                    if isBuild
                        then storePathToFilePath buildInitScript
                        else storePathToFilePath initScript

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

            -- Write network config for init script to read
            when enableNet $ do
                let netConfigContent = T.unlines
                        [ "GUEST_IP=" <> Vm.fnGuestIp netCfg
                        , "GATEWAY=" <> Vm.fnTapIp netCfg
                        , "NETMASK=" <> Vm.fnMask netCfg
                        ]
                -- Ensure /etc exists in rootfs (should already from OCI image)
                mkdirP (rootfsDir </> "etc")
                liftIO $ Prelude.writeFile (rootfsDir </> "etc/network-config") (unpack netConfigContent)

            -- In build mode, write the build command
            when isBuild $ do
                echoErr $ ":: Build command: " <> pack (Prelude.unwords buildCmd)
                writeBuildCmd (rootfsDir </> "build-cmd") buildCmd

            -- Build disk image
            echoErr ":: Building rootfs"
            Vm.buildExt4 rootfsDir disk

            -- Setup TAP networking on host
            when enableNet $ do
                echoErr ":: Setting up TAP network"
                Vm.setupTap netCfg

            -- Boot with appropriate verbosity
            echoErr ":: Booting Firecracker"
            let bootArgs =
                    if isBuild
                        then "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init quiet"
                        else "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init"
            let vmCfg =
                    Vm.defaultFirecrackerConfig
                        { Vm.fcKernel = kernelPath
                        , Vm.fcDisk = disk
                        , Vm.fcCpus = cpus
                        , Vm.fcMemMib = mem
                        , Vm.fcBootArgs = bootArgs
                        , Vm.fcNetwork = if enableNet then Just netCfg else Nothing
                        }

            -- Run VM, then cleanup TAP
            finally
                (Vm.runFirecracker vmCfg)
                (when enableNet $ Vm.teardownTap netCfg)

-- | Write build command script
writeBuildCmd :: FilePath -> [String] -> Sh ()
writeBuildCmd path cmd = do
    let scriptContent =
            T.unlines
                [ "#!/bin/bash"
                , "set -euo pipefail"
                , "cd /root 2>/dev/null || cd /"
                , pack (Prelude.unwords cmd)
                ]
    liftIO $ Prelude.writeFile path (unpack scriptContent)
    run_ "chmod" ["+x", pack path]
