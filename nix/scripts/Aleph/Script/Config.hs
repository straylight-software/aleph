{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Aleph.Script.Config
Description : Type-safe configuration from Dhall

Provides newtypes for Nix-derived values and FromDhall instances.
This is where we get compile-time safety: Dhall's Text becomes
distinct Haskell types that can't be accidentally mixed.

@
-- This won't compile:
runFirecracker kernel rootfs  -- both are StorePath, but...
runFirecracker rootfs kernel  -- oops, swapped! Type error!

-- With proper types:
data FcConfig = FcConfig { fcKernel :: KernelPath, fcRootfs :: RootfsPath }
-- Now Haskell catches the bug at compile time
@

Usage:

@
import Aleph.Script.Config
import Dhall (input, auto)

main = do
  cfg <- input auto "./config.dhall" :: IO FcConfig
  runFirecracker (fcKernel cfg) (fcRootfs cfg)
@
-}
module Aleph.Script.Config (
    -- * Store paths
    StorePath (..),
    DrvPath (..),
    storePathToFilePath,
    drvPathToFilePath,

    -- * System types
    System (..),
    Arch (..),

    -- * Resource types
    MemMiB (..),
    CpuCount (..),
    Port (..),

    -- * Device types
    PciAddress (..),
    DevicePath (..),

    -- * Container types
    ImageRef (..),
    ContainerId (..),

    -- * Config loading
    loadConfig,
    loadConfigFile,
) where

import Data.Text (Text)
import qualified Data.Text as T
import Dhall (FromDhall (..), auto, input, inputFile)
import GHC.Generics (Generic)
import Numeric.Natural (Natural)

-- ============================================================================
-- Store paths
-- ============================================================================

{- | A Nix store path. Newtype ensures you can't accidentally pass
an arbitrary FilePath where a store path is expected.
-}
newtype StorePath = StorePath {unStorePath :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- | A derivation output path (the root, without /bin/foo suffix)
newtype DrvPath = DrvPath {unDrvPath :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- | Convert StorePath to FilePath for use with System.FilePath
storePathToFilePath :: StorePath -> FilePath
storePathToFilePath (StorePath t) = T.unpack t

-- | Convert DrvPath to FilePath
drvPathToFilePath :: DrvPath -> FilePath
drvPathToFilePath (DrvPath t) = T.unpack t

-- ============================================================================
-- System types
-- ============================================================================

-- | System triple (e.g., "x86_64-linux")
newtype System = System {unSystem :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- | CPU architecture (e.g., "x86_64", "aarch64")
newtype Arch = Arch {unArch :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- ============================================================================
-- Resource types
-- ============================================================================

-- | Memory size in MiB
newtype MemMiB = MemMiB {unMemMiB :: Natural}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall, Num)

-- | CPU count
newtype CpuCount = CpuCount {unCpuCount :: Natural}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall, Num)

-- | Network port
newtype Port = Port {unPort :: Natural}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall, Num)

-- ============================================================================
-- Device types
-- ============================================================================

-- | PCI address (e.g., "0000:01:00.0")
newtype PciAddress = PciAddress {unPciAddress :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- | Device path (e.g., "/dev/nvidia0")
newtype DevicePath = DevicePath {unDevicePath :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- ============================================================================
-- Container types
-- ============================================================================

-- | OCI image reference (e.g., "docker.io/library/ubuntu:24.04")
newtype ImageRef = ImageRef {unImageRef :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- | Container ID (typically SHA256)
newtype ContainerId = ContainerId {unContainerId :: Text}
    deriving stock (Eq, Ord, Show, Generic)
    deriving newtype (FromDhall)

-- ============================================================================
-- Config loading utilities
-- ============================================================================

{- | Load a config from a Dhall expression (e.g., embedded in binary)

@
cfg <- loadConfig "{ cpus = 4, mem = 8192 }" :: IO VmConfig
@
-}
loadConfig :: (FromDhall a) => Text -> IO a
loadConfig = input auto

{- | Load a config from a Dhall file

@
cfg <- loadConfigFile "/nix/store/xxx/config.dhall" :: IO VmConfig
@
-}
loadConfigFile :: (FromDhall a) => FilePath -> IO a
loadConfigFile = inputFile auto
