{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Vfio
Description : VFIO/IOMMU device management for GPU passthrough

Type-safe interface to Linux VFIO subsystem. Manages PCI device binding
for GPU passthrough to VMs and containers.

== Sysfs Paths

VFIO operates through sysfs:

@
\/sys\/bus\/pci\/devices\/\<addr\>\/         -- PCI device info
\/sys\/bus\/pci\/devices\/\<addr\>\/vendor   -- e.g., 0x10de (NVIDIA)
\/sys\/bus\/pci\/devices\/\<addr\>\/device   -- e.g., 0x2204 (RTX 3090)
\/sys\/bus\/pci\/devices\/\<addr\>\/driver   -- symlink to current driver
\/sys\/bus\/pci\/devices\/\<addr\>\/iommu_group -- symlink to IOMMU group
\/sys\/bus\/pci\/drivers\/vfio-pci\/bind    -- write addr to bind
\/sys\/bus\/pci\/drivers\/vfio-pci\/unbind  -- write addr to unbind
\/sys\/bus\/pci\/drivers\/vfio-pci\/new_id  -- write "vendor device" to claim
@

== Example

@
import Aleph.Script
import qualified Aleph.Script.Vfio as Vfio

main = script $ do
  -- List all NVIDIA GPUs
  gpus <- Vfio.listNvidiaGpus
  for_ gpus $ \\gpu -> do
    echo $ "Found: " <> Vfio.pciAddr gpu <> " in IOMMU group " <> Vfio.iommuGroup gpu

  -- Bind a GPU to vfio-pci
  Vfio.bindToVfio "0000:01:00.0"

  -- Later, unbind and rescan
  Vfio.unbindFromVfio "0000:01:00.0"
@
-}
module Aleph.Script.Vfio (
    -- * Types
    PciAddr,
    PciDevice (..),
    IommuGroup,

    -- * Discovery
    listNvidiaGpus,
    getPciDevice,
    getIommuGroup,
    getIommuGroupDevices,
    getCurrentDriver,

    -- * Binding
    bindToVfio,
    unbindFromVfio,
    unbindFromCurrentDriver,

    -- * Low-level sysfs
    sysfsRead,
    sysfsWrite,
    sysfsReadlink,
) where

import Aleph.Script hiding (FilePath)
import Control.Monad (forM, forM_)
import qualified Data.Text as T

-- ============================================================================
-- Types
-- ============================================================================

-- | PCI address (e.g., "0000:01:00.0")
type PciAddr = Text

-- | IOMMU group number (e.g., "1")
type IommuGroup = Text

-- | Information about a PCI device
data PciDevice = PciDevice
    { pciAddr :: PciAddr
    -- ^ PCI address
    , pciVendor :: Text
    -- ^ Vendor ID (e.g., "0x10de")
    , pciDevice :: Text
    -- ^ Device ID (e.g., "0x2204")
    , pciDriver :: Maybe Text
    -- ^ Current driver (if bound)
    , pciIommu :: Maybe IommuGroup
    -- ^ IOMMU group (if available)
    , pciDesc :: Maybe Text
    -- ^ Description from lspci
    }
    deriving (Show, Eq)

-- ============================================================================
-- Sysfs Primitives
-- ============================================================================

-- | Base path for PCI devices in sysfs
sysfsPciDevices :: FilePath
sysfsPciDevices = "/sys/bus/pci/devices"

-- | Base path for PCI drivers in sysfs
sysfsPciDrivers :: FilePath
sysfsPciDrivers = "/sys/bus/pci/drivers"

-- | Read a sysfs file, return Nothing on failure
sysfsRead :: FilePath -> Sh (Maybe Text)
sysfsRead path = do
    exists <- test_f path
    if exists
        then Just . strip <$> bash ("cat " <> pack path <> " 2>/dev/null || true")
        else pure Nothing

-- | Write to a sysfs file (requires root)
sysfsWrite :: FilePath -> Text -> Sh ()
sysfsWrite path val = do
    -- Use bash to handle permissions and ignore errors
    bash_ $ "echo '" <> val <> "' > " <> pack path <> " 2>/dev/null || true"

-- | Read a symlink target's basename
sysfsReadlink :: FilePath -> Sh (Maybe Text)
sysfsReadlink path = do
    exists <- test_e path
    if exists
        then do
            target <- bash ("readlink -f " <> pack path <> " 2>/dev/null | xargs basename 2>/dev/null || true")
            let t = strip target
            pure $ if T.null t then Nothing else Just t
        else pure Nothing

-- ============================================================================
-- Discovery
-- ============================================================================

-- | Get information about a specific PCI device
getPciDevice :: PciAddr -> Sh PciDevice
getPciDevice addr = do
    let devPath = sysfsPciDevices </> unpack addr

    vendor <- fromMaybe "unknown" <$> sysfsRead (devPath </> "vendor")
    device <- fromMaybe "unknown" <$> sysfsRead (devPath </> "device")
    driver <- sysfsReadlink (devPath </> "driver")
    iommu <- sysfsReadlink (devPath </> "iommu_group")

    pure
        PciDevice
            { pciAddr = addr
            , pciVendor = vendor
            , pciDevice = device
            , pciDriver = driver
            , pciIommu = iommu
            , pciDesc = Nothing
            }

-- | Get the IOMMU group for a PCI device
getIommuGroup :: PciAddr -> Sh (Maybe IommuGroup)
getIommuGroup addr =
    sysfsReadlink (sysfsPciDevices </> unpack addr </> "iommu_group")

-- | Get all PCI devices in an IOMMU group
getIommuGroupDevices :: IommuGroup -> Sh [PciAddr]
getIommuGroupDevices group = do
    let groupPath = "/sys/kernel/iommu_groups" </> unpack group </> "devices"
    exists <- test_d groupPath
    if exists
        then do
            devs <- bash ("ls " <> pack groupPath <> " 2>/dev/null || true")
            pure $ filter (not . T.null) $ map strip $ T.lines devs
        else pure []

-- | Get the current driver for a PCI device
getCurrentDriver :: PciAddr -> Sh (Maybe Text)
getCurrentDriver addr =
    sysfsReadlink (sysfsPciDevices </> unpack addr </> "driver")

-- | List all NVIDIA GPUs with their info
listNvidiaGpus :: Sh [PciDevice]
listNvidiaGpus = do
    -- Use lspci to find NVIDIA devices
    output <- bash "lspci -D 2>/dev/null | grep -i nvidia || true"
    let lspciLines = filter (not . T.null) $ map strip $ T.lines output

    forM lspciLines $ \line -> do
        let (addrPart, descPart) = T.breakOn " " line
            addr = strip addrPart
            desc = strip $ T.drop 1 descPart

        dev <- getPciDevice addr
        pure dev{pciDesc = Just desc}

-- ============================================================================
-- Binding
-- ============================================================================

-- | Unbind a device from its current driver
unbindFromCurrentDriver :: PciAddr -> Sh ()
unbindFromCurrentDriver addr = do
    mdriver <- getCurrentDriver addr
    case mdriver of
        Nothing -> pure () -- Not bound to anything
        Just driver -> do
            let unbindPath = sysfsPciDrivers </> unpack driver </> "unbind"
            echoErr $ ":: Unbinding " <> addr <> " from " <> driver
            sysfsWrite unbindPath addr

{- | Bind a PCI device (and all devices in its IOMMU group) to vfio-pci

This is the typed equivalent of the bash vfio-bind script.
All IOMMU group members must be bound together.
-}
bindToVfio :: PciAddr -> Sh [PciAddr]
bindToVfio addr = do
    -- Get IOMMU group
    mgroup <- getIommuGroup addr
    group <- case mgroup of
        Nothing -> die $ "Device " <> addr <> " has no IOMMU group"
        Just g -> pure g

    echoErr $ ":: GPU " <> addr <> " is in IOMMU group " <> group

    -- Get all devices in the group
    devices <- getIommuGroupDevices group

    -- Bind each device
    forM_ devices $ \dev -> do
        echoErr $ ":: Binding " <> dev <> " to vfio-pci"

        -- Get vendor/device IDs
        pci <- getPciDevice dev
        let vendorId = T.drop 2 (pciVendor pci) -- Remove "0x" prefix
            deviceId = T.drop 2 (pciDevice pci)

        -- Unbind from current driver
        unbindFromCurrentDriver dev

        -- Register with vfio-pci
        let newIdPath = sysfsPciDrivers </> "vfio-pci" </> "new_id"
            bindPath = sysfsPciDrivers </> "vfio-pci" </> "bind"

        sysfsWrite newIdPath (vendorId <> " " <> deviceId)
        sysfsWrite bindPath dev

    pure devices

{- | Unbind a PCI device (and IOMMU group) from vfio-pci and rescan

This is the typed equivalent of the bash vfio-unbind script.
-}
unbindFromVfio :: PciAddr -> Sh ()
unbindFromVfio addr = do
    -- Get IOMMU group
    mgroup <- getIommuGroup addr
    group <- case mgroup of
        Nothing -> die $ "Device " <> addr <> " has no IOMMU group"
        Just g -> pure g

    -- Get all devices in the group
    devices <- getIommuGroupDevices group

    -- Unbind and remove each device
    forM_ devices $ \dev -> do
        echoErr $ ":: Unbinding " <> dev <> " from vfio-pci"

        let unbindPath = sysfsPciDrivers </> "vfio-pci" </> "unbind"
            removePath = sysfsPciDevices </> unpack dev </> "remove"

        sysfsWrite unbindPath dev
        sysfsWrite removePath "1"

    -- Rescan PCI bus
    echoErr ":: Rescanning PCI bus"
    sysfsWrite "/sys/bus/pci/rescan" "1"
