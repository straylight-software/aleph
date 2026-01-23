-- |
-- Module      : Aleph.Config.Base
-- Description : Base types for Weyl configuration
--
-- Core type aliases for Nix-derived configuration values.
-- These are intentionally simple (Text aliases) because:
--
--   1. Dhall has structural typing - no nominal newtypes
--   2. The real type safety comes from Haskell's newtype wrappers
--   3. Nix generates these configs - it guarantees the invariants
--
-- The 99% bug catch: these become distinct Haskell types via FromDhall,
-- preventing mix-ups at compile time even though Dhall sees them as Text.
--
-- @
-- -- Nix generates:
-- { kernel = "/nix/store/abc123-kernel/vmlinux"
-- , rootfs = "/nix/store/def456-rootfs"
-- }
--
-- -- Haskell reads as distinct types:
-- data Config = Config
--   { kernel :: StorePath  -- newtype StorePath = StorePath Text
--   , rootfs :: StorePath
--   }
-- @

-- ============================================================================
-- Store paths
-- ============================================================================

-- | A Nix store path (e.g., "/nix/store/abc123-foo/bin/foo")
-- Guaranteed by Nix to exist and be immutable.
let StorePath = Text

-- | A derivation output path (store path without /bin/foo suffix)
-- e.g., "/nix/store/abc123-foo"
let DrvPath = Text

-- ============================================================================
-- System types
-- ============================================================================

-- | A system triple (e.g., "x86_64-linux", "aarch64-darwin")
let System = Text

-- | A CPU architecture (e.g., "x86_64", "aarch64")
let Arch = Text

-- ============================================================================
-- Resource types
-- ============================================================================

-- | Memory size in MiB
let MemMiB = Natural

-- | CPU count
let CpuCount = Natural

-- | Port number
let Port = Natural

-- ============================================================================
-- PCI/Device types
-- ============================================================================

-- | PCI address (e.g., "0000:01:00.0")
let PciAddress = Text

-- | Device path (e.g., "/dev/nvidia0")
let DevicePath = Text

-- ============================================================================
-- Container types
-- ============================================================================

-- | OCI image reference (e.g., "docker.io/library/ubuntu:24.04")
let ImageRef = Text

-- | Container ID (SHA256 hash)
let ContainerId = Text

-- ============================================================================
-- Export all types
-- ============================================================================

in  { StorePath
    , DrvPath
    , System
    , Arch
    , MemMiB
    , CpuCount
    , Port
    , PciAddress
    , DevicePath
    , ImageRef
    , ContainerId
    }
