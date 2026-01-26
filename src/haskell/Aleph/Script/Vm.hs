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

== Networking

Firecracker VMs can be configured with TAP networking:

1. Create TAP device on host with 'setupTap'
2. Add 'fcNetwork' to 'FirecrackerConfig'
3. Configure eth0 inside VM (done by init script)
4. Clean up TAP with 'teardownTap'
-}
module Aleph.Script.Vm (
    -- * Rootfs Construction
    injectBusybox,
    buildExt4,
    buildExt4Sized,

    -- * Firecracker
    FirecrackerConfig (..),
    FirecrackerNetwork (..),
    defaultFirecrackerConfig,
    defaultFirecrackerNetwork,
    firecrackerConfigJson,
    runFirecracker,

    -- * TAP Networking
    setupTap,
    teardownTap,

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

-- | Firecracker network configuration
data FirecrackerNetwork = FirecrackerNetwork
    { fnTapDev :: Text
    -- ^ TAP device name (e.g., "tap0")
    , fnTapIp :: Text
    -- ^ Host IP for TAP (e.g., "172.16.0.1")
    , fnGuestIp :: Text
    -- ^ Guest IP (e.g., "172.16.0.2")
    , fnGuestMac :: Text
    -- ^ Guest MAC address (e.g., "06:00:AC:10:00:02")
    , fnMask :: Text
    -- ^ Network mask (e.g., "/30")
    }
    deriving (Show)

-- | Default network config using 172.16.0.0/30 subnet
defaultFirecrackerNetwork :: FirecrackerNetwork
defaultFirecrackerNetwork =
    FirecrackerNetwork
        { fnTapDev = "fctap0"
        , fnTapIp = "172.16.0.1"
        , fnGuestIp = "172.16.0.2"
        , fnGuestMac = "06:00:AC:10:00:02"
        , fnMask = "/30"
        }

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
    , fcNetwork :: Maybe FirecrackerNetwork
    -- ^ Optional network configuration
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
        , fcNetwork = Nothing
        }

-- | Generate Firecracker config JSON
firecrackerConfigJson :: FirecrackerConfig -> Sh Text
firecrackerConfigJson FirecrackerConfig{..} = do
    -- Use jq to build the JSON (handles escaping correctly)
    -- nullInput = True (-n) because we're not reading from any file
    let baseArgs =
            [ Jq.arg "kernel" (pack fcKernel)
            , Jq.arg "disk" (pack fcDisk)
            , Jq.arg "boot_args" fcBootArgs
            , Jq.argjson "cpus" (pack $ show fcCpus)
            , Jq.argjson "mem" (pack $ show fcMemMib)
            ]
        -- Add network args if configured
        netArgs = case fcNetwork of
            Nothing -> []
            Just FirecrackerNetwork{..} ->
                [ Jq.arg "tap_dev" fnTapDev
                , Jq.arg "guest_mac" fnGuestMac
                ]
        allArgs = baseArgs ++ netArgs
        -- Base config without network
        baseFilter =
            "{ \"boot-source\": { \"kernel_image_path\": $kernel, \"boot_args\": $boot_args }, "
                <> "\"drives\": [{ \"drive_id\": \"rootfs\", \"path_on_host\": $disk, \"is_root_device\": true, \"is_read_only\": false }], "
                <> "\"machine-config\": { \"vcpu_count\": $cpus, \"mem_size_mib\": $mem }"
        -- Add network-interfaces if configured
        filterWithNet = case fcNetwork of
            Nothing -> baseFilter <> " }"
            Just _ ->
                baseFilter
                    <> ", \"network-interfaces\": [{ \"iface_id\": \"eth0\", \"guest_mac\": $guest_mac, \"host_dev_name\": $tap_dev }] }"
    Jq.jqWithArgs
        Jq.defaults{Jq.nullInput = True}
        allArgs
        filterWithNet
        []

-- | Setup TAP device on host for VM networking
--
-- Creates TAP device, assigns IP, enables forwarding and NAT.
-- Requires root/sudo. Returns the TAP device name.
setupTap :: FirecrackerNetwork -> Sh ()
setupTap FirecrackerNetwork{..} = do
    -- Delete existing TAP if present (ignore errors)
    errExit False $ run_ "sudo" ["ip", "link", "del", fnTapDev]

    -- Create TAP device
    run_ "sudo" ["ip", "tuntap", "add", "dev", fnTapDev, "mode", "tap"]

    -- Assign IP address
    run_ "sudo" ["ip", "addr", "add", fnTapIp <> fnMask, "dev", fnTapDev]

    -- Bring up the interface
    run_ "sudo" ["ip", "link", "set", "dev", fnTapDev, "up"]

    -- Enable IP forwarding
    run_ "sudo" ["sh", "-c", "echo 1 > /proc/sys/net/ipv4/ip_forward"]

    -- Get default route interface for NAT
    hostIface <-
        run "ip" ["-j", "route", "list", "default"]
            & silently
    let ifaceName = extractDefaultIface hostIface

    -- Setup NAT masquerading (delete first to avoid duplicates)
    errExit False $ run_ "sudo" ["iptables", "-t", "nat", "-D", "POSTROUTING", "-o", ifaceName, "-j", "MASQUERADE"]
    run_ "sudo" ["iptables", "-t", "nat", "-A", "POSTROUTING", "-o", ifaceName, "-j", "MASQUERADE"]

    -- Allow forwarding
    run_ "sudo" ["iptables", "-P", "FORWARD", "ACCEPT"]
  where
    -- Extract interface name from `ip -j route list default` output
    -- Returns "eth0" as fallback if parsing fails
    extractDefaultIface :: Text -> Text
    extractDefaultIface json =
        -- Simple extraction: look for "dev":"xxx" pattern
        case T.breakOn "\"dev\":\"" json of
            (_, rest)
                | T.null rest -> "eth0"
                | otherwise ->
                    let afterDev = T.drop 7 rest -- drop "dev":"
                     in T.takeWhile (/= '"') afterDev

-- | Teardown TAP device
teardownTap :: FirecrackerNetwork -> Sh ()
teardownTap FirecrackerNetwork{..} = do
    errExit False $ run_ "sudo" ["ip", "link", "del", fnTapDev]
    pure ()

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
