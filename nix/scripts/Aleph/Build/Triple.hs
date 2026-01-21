{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}

-- | Aleph.Build.Triple
-- Real target triples. Sum types, not strings.
module Aleph.Build.Triple
  ( -- Types
    Arch (..)
  , Vendor (..)
  , OS (..)
  , ABI (..)
  , Triple (..)

    -- Conversion
  , toString
  , parse

    -- Predicates
  , isLinux
  , isDarwin
  , isWindows
  , isWasm
  , isCross

    -- Standard triples
  , x86_64_linux_gnu
  , x86_64_linux_musl
  , aarch64_linux_gnu
  , aarch64_linux_musl
  , x86_64_darwin
  , aarch64_darwin
  , wasm32_wasi

    -- LLVM target names
  , llvmArch
  ) where

import Data.List (intercalate)

-- | CPU architecture
data Arch
  = X86_64
  | AArch64
  | ARMv7
  | RISCV64
  | WASM32
  | PowerPC64LE
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Vendor
data Vendor
  = Unknown
  | Apple
  | PC
  | W64
  | Nvidia
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Operating system
data OS
  = Linux
  | Darwin
  | Windows
  | WASI
  | NoOS
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | ABI
data ABI
  = GNU
  | Musl
  | MSVC
  | EABI
  | Android
  | NoABI
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Complete target triple
data Triple = Triple
  { arch :: Arch
  , vendor :: Vendor
  , os :: OS
  , abi :: ABI
  }
  deriving (Show, Eq, Ord)

--------------------------------------------------------------------------------
-- Conversion
--------------------------------------------------------------------------------

toString :: Triple -> String
toString Triple {..} =
  intercalate "-" $
    filter (not . null) [archStr arch, vendorStr vendor, osStr os, abiStr abi]
  where
    archStr = \case
      X86_64 -> "x86_64"
      AArch64 -> "aarch64"
      ARMv7 -> "armv7"
      RISCV64 -> "riscv64"
      WASM32 -> "wasm32"
      PowerPC64LE -> "powerpc64le"

    vendorStr = \case
      Unknown -> "unknown"
      Apple -> "apple"
      PC -> "pc"
      W64 -> "w64"
      Nvidia -> "nvidia"

    osStr = \case
      Linux -> "linux"
      Darwin -> "darwin"
      Windows -> "windows"
      WASI -> "wasi"
      NoOS -> "none"

    abiStr = \case
      GNU -> "gnu"
      Musl -> "musl"
      MSVC -> "msvc"
      EABI -> "eabi"
      Android -> "android"
      NoABI -> ""

-- | Parse a triple string (simplified)
parse :: String -> Maybe Triple
parse s = case words $ map dashToSpace s of
  [a, v, o] -> Triple <$> parseArch a <*> parseVendor v <*> parseOS o <*> pure NoABI
  [a, v, o, ab] -> Triple <$> parseArch a <*> parseVendor v <*> parseOS o <*> parseABI ab
  _ -> Nothing
  where
    dashToSpace '-' = ' '
    dashToSpace c = c

    parseArch = \case
      "x86_64" -> Just X86_64
      "aarch64" -> Just AArch64
      "armv7" -> Just ARMv7
      "riscv64" -> Just RISCV64
      "wasm32" -> Just WASM32
      "powerpc64le" -> Just PowerPC64LE
      _ -> Nothing

    parseVendor = \case
      "unknown" -> Just Unknown
      "apple" -> Just Apple
      "pc" -> Just PC
      "w64" -> Just W64
      "nvidia" -> Just Nvidia
      _ -> Nothing

    parseOS = \case
      "linux" -> Just Linux
      "darwin" -> Just Darwin
      "windows" -> Just Windows
      "wasi" -> Just WASI
      "none" -> Just NoOS
      _ -> Nothing

    parseABI = \case
      "gnu" -> Just GNU
      "musl" -> Just Musl
      "msvc" -> Just MSVC
      "eabi" -> Just EABI
      "android" -> Just Android
      "" -> Just NoABI
      _ -> Nothing

--------------------------------------------------------------------------------
-- Predicates
--------------------------------------------------------------------------------

isLinux :: Triple -> Bool
isLinux t = os t == Linux

isDarwin :: Triple -> Bool
isDarwin t = os t == Darwin

isWindows :: Triple -> Bool
isWindows t = os t == Windows

isWasm :: Triple -> Bool
isWasm t = arch t == WASM32

isCross :: Triple -> Triple -> Bool
isCross host target = host /= target

--------------------------------------------------------------------------------
-- Standard triples
--------------------------------------------------------------------------------

x86_64_linux_gnu :: Triple
x86_64_linux_gnu = Triple X86_64 Unknown Linux GNU

x86_64_linux_musl :: Triple
x86_64_linux_musl = Triple X86_64 Unknown Linux Musl

aarch64_linux_gnu :: Triple
aarch64_linux_gnu = Triple AArch64 Unknown Linux GNU

aarch64_linux_musl :: Triple
aarch64_linux_musl = Triple AArch64 Unknown Linux Musl

x86_64_darwin :: Triple
x86_64_darwin = Triple X86_64 Apple Darwin NoABI

aarch64_darwin :: Triple
aarch64_darwin = Triple AArch64 Apple Darwin NoABI

wasm32_wasi :: Triple
wasm32_wasi = Triple WASM32 Unknown WASI NoABI

--------------------------------------------------------------------------------
-- LLVM
--------------------------------------------------------------------------------

-- | LLVM target name for architecture
llvmArch :: Arch -> String
llvmArch = \case
  X86_64 -> "X86"
  AArch64 -> "AArch64"
  ARMv7 -> "ARM"
  RISCV64 -> "RISCV"
  WASM32 -> "WebAssembly"
  PowerPC64LE -> "PowerPC"
