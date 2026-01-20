{-# LANGUAGE OverloadedStrings #-}

{- |
Run a build command inside an OCI container in a Firecracker microVM.

Usage: fc-build IMAGE COMMAND...

Environment:
  FC_CPUS  - Number of vCPUs (default: 2)
  FC_MEM   - Memory in MiB (default: 1024)

Example: fc-build ubuntu:24.04 make -j8
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Vm as Vm
import qualified Data.List as L
import System.Environment (getArgs, lookupEnv)
import Text.Read (readMaybe)

-- Nix-substituted paths
fcKernelPath :: FilePath
fcKernelPath = "/nix/store/PLACEHOLDER-fc-kernel/vmlinux"

busyboxPath :: FilePath
busyboxPath = "/nix/store/PLACEHOLDER-busybox/bin/busybox"

fcBuildInitScript :: FilePath
fcBuildInitScript = "/nix/store/PLACEHOLDER-fc-build-init/init"

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> script $ do
            echoErr "Usage: fc-build IMAGE COMMAND..."
            echoErr ""
            echoErr "Run a build command in a Firecracker microVM."
            echoErr ""
            echoErr "Example: fc-build ubuntu:24.04 make -j8"
            exit 1
        (image : cmdArgs) -> do
            when (L.null cmdArgs) $ do
                putStrLn "Error: COMMAND required"
                script $ exit 1

            cpus <- maybe 2 id . (>>= readMaybe) <$> lookupEnv "FC_CPUS"
            mem <- maybe 1024 id . (>>= readMaybe) <$> lookupEnv "FC_MEM"

            let buildCmd = L.unwords cmdArgs

            script $ do
                echoErr $ ":: Firecracker Build VM (" <> pack (show cpus) <> " CPUs, " <> pack (show mem) <> " MiB)"
                echoErr $ ":: Command: " <> pack buildCmd

                withTmpDir $ \workDir -> do
                    let rootfs = workDir </> "rootfs"
                        disk = workDir </> "disk.ext4"

                    -- Pull image
                    echoErr $ ":: Pulling " <> pack image
                    mkdirP rootfs
                    setEnv "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt"
                    bash_ $ "crane export --platform linux/amd64 '" <> pack image <> "' - | tar -xf - -C " <> pack rootfs

                    -- Inject busybox and init
                    echoErr ":: Injecting init + build command"
                    Vm.injectBusybox busyboxPath rootfs
                    cp fcBuildInitScript (rootfs </> "init")
                    run_ "chmod" ["+x", pack (rootfs </> "init")]

                    -- Write build command
                    liftIO $
                        writeFile (rootfs </> "build-cmd") $
                            "#!/bin/sh\ncd /root\nexec " ++ buildCmd ++ "\n"
                    run_ "chmod" ["+x", pack (rootfs </> "build-cmd")]

                    -- Build disk image
                    echoErr ":: Building rootfs"
                    Vm.buildExt4 rootfs disk

                    -- Boot
                    echoErr ":: Booting Firecracker"
                    let cfg =
                            Vm.defaultFirecrackerConfig
                                { Vm.fcKernel = fcKernelPath
                                , Vm.fcDisk = disk
                                , Vm.fcCpus = cpus
                                , Vm.fcMemMib = mem
                                , Vm.fcBootArgs = "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init quiet"
                                }
                    Vm.runFirecracker cfg
  where
    when cond action = if cond then action else pure ()
