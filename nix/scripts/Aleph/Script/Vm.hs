{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Vm
Description : VM rootfs construction for Firecracker and Cloud Hypervisor

Shared infrastructure for building VM disk images from OCI containers.

== Workflow

1. Pull OCI image with 'Aleph.Script.Oci.pullOrCache'
2. Inject busybox and init script with 'injectInit'
3. Build ext4 disk image with 'buildExt4'
4. Generate VM config and launch
-}
module Aleph.Script.Vm (
    -- * Rootfs Construction
    injectBusybox,
    buildExt4,
    buildExt4Sized,

    -- * Firecracker
    FirecrackerConfig (..),
    defaultFirecrackerConfig,
    firecrackerConfigJson,
    runFirecracker,

    -- * Cloud Hypervisor
    CloudHypervisorConfig (..),
    defaultCloudHypervisorConfig,
    runCloudHypervisor,
    runCloudHypervisorGpu,
) where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Tools.Jq as Jq
import qualified Aleph.Script.Vfio as Vfio
import Control.Monad (forM_)
import Data.Function ((&))
import qualified Data.Text as T

-- ============================================================================
-- Rootfs Construction
-- ============================================================================

-- | Busybox commands to symlink
busyboxCmds :: [Text]
busyboxCmds =
    [ "sh"
    , "ash"
    , "awk"
    , "cat"
    , "chmod"
    , "chown"
    , "clear"
    , "cp"
    , "cttyhack"
    , "dd"
    , "df"
    , "dmesg"
    , "echo"
    , "env"
    , "free"
    , "grep"
    , "head"
    , "hostname"
    , "id"
    , "ifconfig"
    , "ip"
    , "kill"
    , "ln"
    , "ls"
    , "mkdir"
    , "mknod"
    , "modprobe"
    , "mount"
    , "mv"
    , "nproc"
    , "ping"
    , "poweroff"
    , "ps"
    , "pwd"
    , "reboot"
    , "rm"
    , "sed"
    , "seq"
    , "setsid"
    , "sleep"
    , "stat"
    , "sync"
    , "tail"
    , "tar"
    , "touch"
    , "uname"
    , "umount"
    , "vi"
    , "wc"
    , "which"
    ]

{- | Inject busybox into rootfs for minimal shell support

Creates /usr/local/bin/busybox and symlinks for common commands.
Also creates /bin/sh symlink for scripts with #!/bin/sh shebang.
Requires busybox path to be passed in (from Nix).
-}
injectBusybox :: FilePath -> FilePath -> Sh ()
injectBusybox busyboxPath rootfs = do
    let binDir = rootfs </> "usr/local/bin"
    mkdirP binDir
    cp busyboxPath (binDir </> "busybox")

    -- Create symlinks for each command
    forM_ busyboxCmds $ \cmd -> do
        let linkPath = binDir </> unpack cmd
        -- Ignore errors if link exists
        errExit False $ symlink "busybox" linkPath
        pure ()

    -- Create /bin/sh symlink for scripts with #!/bin/sh shebang
    -- This is critical for init scripts to work
    let rootBin = rootfs </> "bin"
    mkdirP rootBin
    errExit False $ symlink "/usr/local/bin/sh" (rootBin </> "sh")
    pure ()

{- | Build ext4 disk image from rootfs directory

Uses genext2fs + tune2fs to create a bootable ext4 image.
-}
buildExt4 :: FilePath -> FilePath -> Sh ()
buildExt4 = buildExt4Sized (4 * 1024 * 1024) -- 4GB default

-- | Build ext4 disk image with specific size (in 1K blocks)
buildExt4Sized :: Int -> FilePath -> FilePath -> Sh ()
buildExt4Sized sizeBlocks rootfs diskPath = do
    -- genext2fs creates ext2, tune2fs upgrades to ext4
    run_
        "fakeroot"
        [ "genext2fs"
        , "-B"
        , "4096" -- 4K block size
        , "-b"
        , pack (show sizeBlocks)
        , "-d"
        , pack rootfs
        , pack diskPath
        ]

    -- Upgrade to ext4 with modern features
    errExit False $
        run_
            "tune2fs"
            [ "-O"
            , "extents,uninit_bg,dir_index,has_journal"
            , pack diskPath
            ]
    pure ()

-- ============================================================================
-- Firecracker
-- ============================================================================

-- | Firecracker VM configuration
data FirecrackerConfig = FirecrackerConfig
    { fcKernel :: FilePath
    -- ^ Path to vmlinux kernel
    , fcDisk :: FilePath
    -- ^ Path to rootfs disk image
    , fcCpus :: Int
    -- ^ Number of vCPUs
    , fcMemMib :: Int
    -- ^ Memory in MiB
    , fcBootArgs :: Text
    -- ^ Kernel command line
    }
    deriving (Show)

-- | Default Firecracker config
defaultFirecrackerConfig :: FirecrackerConfig
defaultFirecrackerConfig =
    FirecrackerConfig
        { fcKernel = ""
        , fcDisk = ""
        , fcCpus = 2
        , fcMemMib = 1024
        , fcBootArgs = "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init"
        }

-- | Generate Firecracker config JSON
firecrackerConfigJson :: FirecrackerConfig -> Sh Text
firecrackerConfigJson FirecrackerConfig{..} = do
    -- Use jq to build the JSON (handles escaping correctly)
    -- nullInput = True (-n) because we're not reading from any file
    Jq.jqWithArgs
        Jq.defaults{Jq.nullInput = True}
        [ Jq.arg "kernel" (pack fcKernel)
        , Jq.arg "disk" (pack fcDisk)
        , Jq.arg "boot_args" fcBootArgs
        , Jq.argjson "cpus" (pack $ show fcCpus)
        , Jq.argjson "mem" (pack $ show fcMemMib)
        ]
        ( "{ \"boot-source\": { \"kernel_image_path\": $kernel, \"boot_args\": $boot_args }, "
            <> "\"drives\": [{ \"drive_id\": \"rootfs\", \"path_on_host\": $disk, \"is_root_device\": true, \"is_read_only\": false }], "
            <> "\"machine-config\": { \"vcpu_count\": $cpus, \"mem_size_mib\": $mem } }"
        )
        []

-- | Run Firecracker with config
runFirecracker :: FirecrackerConfig -> Sh ()
runFirecracker cfg = do
    configJson <- firecrackerConfigJson cfg

    -- Write config to temp file and run
    withTmpFile $ \configPath -> do
        liftIO $ writeFile configPath (unpack configJson)
        run_ "firecracker" ["--no-api", "--config-file", pack configPath]

-- ============================================================================
-- Cloud Hypervisor
-- ============================================================================

-- | Cloud Hypervisor VM configuration
data CloudHypervisorConfig = CloudHypervisorConfig
    { chKernel :: FilePath
    -- ^ Path to vmlinux kernel
    , chDisk :: FilePath
    -- ^ Path to rootfs disk image
    , chCpus :: Int
    -- ^ Number of vCPUs
    , chMemMib :: Int
    -- ^ Memory in MiB
    , chBootArgs :: Text
    -- ^ Kernel command line
    , chConsole :: Text
    -- ^ Console type ("tty" or "off")
    }
    deriving (Show)

-- | Default Cloud Hypervisor config
defaultCloudHypervisorConfig :: CloudHypervisorConfig
defaultCloudHypervisorConfig =
    CloudHypervisorConfig
        { chKernel = ""
        , chDisk = ""
        , chCpus = 2
        , chMemMib = 1024
        , chBootArgs = "console=ttyS0 root=/dev/vda rw init=/init"
        , chConsole = "tty"
        }

-- | Run Cloud Hypervisor
runCloudHypervisor :: CloudHypervisorConfig -> Sh ()
runCloudHypervisor CloudHypervisorConfig{..} = do
    run_
        "cloud-hypervisor"
        [ "--kernel"
        , pack chKernel
        , "--disk"
        , "path=" <> pack chDisk
        , "--cpus"
        , "boot=" <> pack (show chCpus)
        , "--memory"
        , "size=" <> pack (show chMemMib) <> "M"
        , "--cmdline"
        , chBootArgs
        , "--console"
        , chConsole
        , "--serial"
        , "tty"
        ]

-- | Run Cloud Hypervisor with GPU passthrough
runCloudHypervisorGpu :: CloudHypervisorConfig -> Vfio.PciAddr -> Sh ()
runCloudHypervisorGpu cfg@CloudHypervisorConfig{..} gpuAddr = do
    -- Get IOMMU group devices
    mgroup <- Vfio.getIommuGroup gpuAddr
    devices <- case mgroup of
        Nothing -> die $ "GPU " <> gpuAddr <> " has no IOMMU group"
        Just g -> Vfio.getIommuGroupDevices g

    -- Build device arguments
    let deviceArgs = concatMap (\d -> ["--device", "path=/sys/bus/pci/devices/" <> d]) devices

    run_ "cloud-hypervisor" $
        [ "--kernel"
        , pack chKernel
        , "--disk"
        , "path=" <> pack chDisk
        , "--cpus"
        , "boot=" <> pack (show chCpus)
        , "--memory"
        , "size=" <> pack (show chMemMib) <> "M"
        , "--cmdline"
        , chBootArgs
        , "--console"
        , chConsole
        , "--serial"
        , "tty"
        ]
            ++ deviceArgs
