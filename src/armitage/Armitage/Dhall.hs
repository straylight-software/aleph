{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Dhall
Description : Dhall configuration parsing for build targets

Parse build targets from Dhall files. This connects:
  - src/armitage/dhall/*.dhall (type definitions)
  - BUILD.dhall files in projects (target definitions)
  - Armitage.Builder (execution)

The Dhall types mirror the Haskell types exactly, so parsing is
just a matter of calling 'inputFile auto'.

Usage:
  target <- loadTarget "BUILD.dhall"
  result <- runBuild config (targetToDerivation target)
-}
module Armitage.Dhall
  ( -- * Target Types (from Dhall)
    Target (..)
  , Src (..)
  , Dep (..)
  , Toolchain (..)
  , Compiler (..)
  , Triple (..)
  , Arch (..)
  , Vendor (..)
  , OS (..)
  , ABI (..)
  , Cpu (..)
  , Gpu (..)
  , CFlag (..)
  , OptLevel (..)
  , LTOMode (..)
  , CStd (..)
  , CxxStd (..)
  , Sanitizer (..)
  , DebugLevel (..)
  , LDFlag (..)
  , Resource (..)

    -- * Loading
  , loadTarget
  , loadTargets
  , loadToolchain

    -- * Conversion
  , targetToDerivation
  , resourceToCoeffect

    -- * Rendering
  , renderTriple
  , renderCpu
  , renderGpu
  , renderCFlag
  , renderLDFlag
  ) where

import Control.Exception (throwIO)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Dhall (FromDhall (..), Decoder, inputFile, auto, constructor, union, field, record, Natural, unit)
import GHC.Generics (Generic)

import qualified Armitage.Builder as Builder

-- -----------------------------------------------------------------------------
-- Architecture Types (mirror src/armitage/dhall/Triple.dhall)
-- Dhall uses lowercase constructors, so we match exactly
-- -----------------------------------------------------------------------------

data Arch = Arch_x86_64 | Arch_aarch64 | Arch_riscv64 | Arch_wasm32 | Arch_armv7
  deriving stock (Show, Eq, Generic)

instance FromDhall Arch where
  autoWith _ = union
    ( (Arch_x86_64 <$ constructor "x86_64" unit)
    <> (Arch_aarch64 <$ constructor "aarch64" unit)
    <> (Arch_riscv64 <$ constructor "riscv64" unit)
    <> (Arch_wasm32 <$ constructor "wasm32" unit)
    <> (Arch_armv7 <$ constructor "armv7" unit)
    )

data Vendor = Vendor_unknown | Vendor_pc | Vendor_apple | Vendor_nvidia
  deriving stock (Show, Eq, Generic)

instance FromDhall Vendor where
  autoWith _ = union
    ( (Vendor_unknown <$ constructor "unknown" unit)
    <> (Vendor_pc <$ constructor "pc" unit)
    <> (Vendor_apple <$ constructor "apple" unit)
    <> (Vendor_nvidia <$ constructor "nvidia" unit)
    )

data OS = OS_linux | OS_darwin | OS_windows | OS_wasi | OS_none
  deriving stock (Show, Eq, Generic)

instance FromDhall OS where
  autoWith _ = union
    ( (OS_linux <$ constructor "linux" unit)
    <> (OS_darwin <$ constructor "darwin" unit)
    <> (OS_windows <$ constructor "windows" unit)
    <> (OS_wasi <$ constructor "wasi" unit)
    <> (OS_none <$ constructor "none" unit)
    )

data ABI = ABI_gnu | ABI_musl | ABI_eabi | ABI_eabihf | ABI_msvc | ABI_none
  deriving stock (Show, Eq, Generic)

instance FromDhall ABI where
  autoWith _ = union
    ( (ABI_gnu <$ constructor "gnu" unit)
    <> (ABI_musl <$ constructor "musl" unit)
    <> (ABI_eabi <$ constructor "eabi" unit)
    <> (ABI_eabihf <$ constructor "eabihf" unit)
    <> (ABI_msvc <$ constructor "msvc" unit)
    <> (ABI_none <$ constructor "none" unit)
    )

data Cpu
  = Cpu_generic | Cpu_native
  -- x86_64
  | Cpu_x86_64_v2 | Cpu_x86_64_v3 | Cpu_x86_64_v4
  | Cpu_znver3 | Cpu_znver4 | Cpu_znver5 | Cpu_sapphirerapids | Cpu_alderlake
  -- aarch64 datacenter
  | Cpu_neoverse_v2 | Cpu_neoverse_n2
  -- aarch64 embedded
  | Cpu_cortex_a78ae | Cpu_cortex_a78c
  -- aarch64 consumer
  | Cpu_apple_m1 | Cpu_apple_m2 | Cpu_apple_m3 | Cpu_apple_m4
  deriving stock (Show, Eq, Generic)

instance FromDhall Cpu where
  autoWith _ = union
    ( (Cpu_generic <$ constructor "generic" unit)
    <> (Cpu_native <$ constructor "native" unit)
    <> (Cpu_x86_64_v2 <$ constructor "x86_64_v2" unit)
    <> (Cpu_x86_64_v3 <$ constructor "x86_64_v3" unit)
    <> (Cpu_x86_64_v4 <$ constructor "x86_64_v4" unit)
    <> (Cpu_znver3 <$ constructor "znver3" unit)
    <> (Cpu_znver4 <$ constructor "znver4" unit)
    <> (Cpu_znver5 <$ constructor "znver5" unit)
    <> (Cpu_sapphirerapids <$ constructor "sapphirerapids" unit)
    <> (Cpu_alderlake <$ constructor "alderlake" unit)
    <> (Cpu_neoverse_v2 <$ constructor "neoverse_v2" unit)
    <> (Cpu_neoverse_n2 <$ constructor "neoverse_n2" unit)
    <> (Cpu_cortex_a78ae <$ constructor "cortex_a78ae" unit)
    <> (Cpu_cortex_a78c <$ constructor "cortex_a78c" unit)
    <> (Cpu_apple_m1 <$ constructor "apple_m1" unit)
    <> (Cpu_apple_m2 <$ constructor "apple_m2" unit)
    <> (Cpu_apple_m3 <$ constructor "apple_m3" unit)
    <> (Cpu_apple_m4 <$ constructor "apple_m4" unit)
    )

data Gpu
  = Gpu_none
  -- Ampere
  | Gpu_sm_80 | Gpu_sm_86
  -- Ada Lovelace
  | Gpu_sm_89
  -- Hopper
  | Gpu_sm_90 | Gpu_sm_90a
  -- Orin
  | Gpu_sm_87
  -- Blackwell
  | Gpu_sm_100 | Gpu_sm_100a | Gpu_sm_120
  deriving stock (Show, Eq, Generic)

instance FromDhall Gpu where
  autoWith _ = union
    ( (Gpu_none <$ constructor "none" unit)
    <> (Gpu_sm_80 <$ constructor "sm_80" unit)
    <> (Gpu_sm_86 <$ constructor "sm_86" unit)
    <> (Gpu_sm_87 <$ constructor "sm_87" unit)
    <> (Gpu_sm_89 <$ constructor "sm_89" unit)
    <> (Gpu_sm_90 <$ constructor "sm_90" unit)
    <> (Gpu_sm_90a <$ constructor "sm_90a" unit)
    <> (Gpu_sm_100 <$ constructor "sm_100" unit)
    <> (Gpu_sm_100a <$ constructor "sm_100a" unit)
    <> (Gpu_sm_120 <$ constructor "sm_120" unit)
    )

data Triple = Triple
  { arch :: Arch
  , vendor :: Vendor
  , os :: OS
  , abi :: ABI
  , cpu :: Cpu
  , gpu :: Gpu
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (FromDhall)

-- -----------------------------------------------------------------------------
-- Compiler Types (mirror src/armitage/dhall/Toolchain.dhall)
-- -----------------------------------------------------------------------------

data Compiler
  = Compiler_Clang { version :: Text }
  | Compiler_NVClang { version :: Text }
  | Compiler_GCC { version :: Text }
  | Compiler_NVCC { version :: Text }
  | Compiler_Rustc { version :: Text }
  | Compiler_GHC { version :: Text }
  | Compiler_Lean { version :: Text }
  deriving stock (Show, Eq, Generic)

instance FromDhall Compiler where
  autoWith opts = union
    ( constructor "Clang" (record (Compiler_Clang <$> field "version" auto))
    <> constructor "NVClang" (record (Compiler_NVClang <$> field "version" auto))
    <> constructor "GCC" (record (Compiler_GCC <$> field "version" auto))
    <> constructor "NVCC" (record (Compiler_NVCC <$> field "version" auto))
    <> constructor "Rustc" (record (Compiler_Rustc <$> field "version" auto))
    <> constructor "GHC" (record (Compiler_GHC <$> field "version" auto))
    <> constructor "Lean" (record (Compiler_Lean <$> field "version" auto))
    )

-- -----------------------------------------------------------------------------
-- Flag Types (mirror src/armitage/dhall/CFlags.dhall, LDFlags.dhall)
-- -----------------------------------------------------------------------------

data OptLevel = O0 | O1 | O2 | O3 | Os | Oz | Og
  deriving stock (Show, Eq, Generic)

instance FromDhall OptLevel where
  autoWith _ = union
    ( (O0 <$ constructor "O0" unit)
    <> (O1 <$ constructor "O1" unit)
    <> (O2 <$ constructor "O2" unit)
    <> (O3 <$ constructor "O3" unit)
    <> (Os <$ constructor "Os" unit)
    <> (Oz <$ constructor "Oz" unit)
    <> (Og <$ constructor "Og" unit)
    )

data LTOMode = LTO_off | LTO_thin | LTO_full
  deriving stock (Show, Eq, Generic)

instance FromDhall LTOMode where
  autoWith _ = union
    ( (LTO_off <$ constructor "off" unit)
    <> (LTO_thin <$ constructor "thin" unit)
    <> (LTO_full <$ constructor "full" unit)
    )

data CStd = C89 | C99 | C11 | C17 | C23
  deriving stock (Show, Eq, Generic)

instance FromDhall CStd where
  autoWith _ = union
    ( (C89 <$ constructor "c89" unit)
    <> (C99 <$ constructor "c99" unit)
    <> (C11 <$ constructor "c11" unit)
    <> (C17 <$ constructor "c17" unit)
    <> (C23 <$ constructor "c23" unit)
    )

data CxxStd = Cxx11 | Cxx14 | Cxx17 | Cxx20 | Cxx23
  deriving stock (Show, Eq, Generic)

instance FromDhall CxxStd where
  autoWith _ = union
    ( (Cxx11 <$ constructor "cxx11" unit)
    <> (Cxx14 <$ constructor "cxx14" unit)
    <> (Cxx17 <$ constructor "cxx17" unit)
    <> (Cxx20 <$ constructor "cxx20" unit)
    <> (Cxx23 <$ constructor "cxx23" unit)
    )

data Sanitizer = San_address | San_memory | San_thread | San_undefined | San_leak
  deriving stock (Show, Eq, Generic)

instance FromDhall Sanitizer where
  autoWith _ = union
    ( (San_address <$ constructor "address" unit)
    <> (San_memory <$ constructor "memory" unit)
    <> (San_thread <$ constructor "thread" unit)
    <> (San_undefined <$ constructor "undefined" unit)
    <> (San_leak <$ constructor "leak" unit)
    )

data DebugLevel = G0 | G1 | G2 | G3
  deriving stock (Show, Eq, Generic)

instance FromDhall DebugLevel where
  autoWith _ = union
    ( (G0 <$ constructor "g0" unit)
    <> (G1 <$ constructor "g1" unit)
    <> (G2 <$ constructor "g2" unit)
    <> (G3 <$ constructor "g3" unit)
    )

-- | C/C++ compiler flags (matches CFlags.dhall CFlag union)
data CFlag
  = CFlag_Opt OptLevel
  | CFlag_LTO LTOMode
  | CFlag_StdC CStd
  | CFlag_StdCxx CxxStd
  | CFlag_Wall
  | CFlag_Wextra
  | CFlag_Werror
  | CFlag_Wpedantic
  | CFlag_Wno Text
  | CFlag_Define { name :: Text, value :: Maybe Text }
  | CFlag_Undef Text
  | CFlag_Include Text
  | CFlag_System Text
  | CFlag_PIC
  | CFlag_PIE
  | CFlag_Static
  | CFlag_Shared
  | CFlag_March Text
  | CFlag_Mtune Text
  | CFlag_Native
  | CFlag_Debug DebugLevel
  | CFlag_Sanitize Sanitizer
  | CFlag_FunctionSections
  | CFlag_DataSections
  | CFlag_NoExceptions
  | CFlag_NoRTTI
  | CFlag_Pthread
  | CFlag_Raw Text
  deriving stock (Show, Eq, Generic)

instance FromDhall CFlag where
  autoWith opts = union
    ( constructor "Opt" (CFlag_Opt <$> autoWith opts)
    <> constructor "LTO" (CFlag_LTO <$> autoWith opts)
    <> constructor "StdC" (CFlag_StdC <$> autoWith opts)
    <> constructor "StdCxx" (CFlag_StdCxx <$> autoWith opts)
    <> (CFlag_Wall <$ constructor "Wall" unit)
    <> (CFlag_Wextra <$ constructor "Wextra" unit)
    <> (CFlag_Werror <$ constructor "Werror" unit)
    <> (CFlag_Wpedantic <$ constructor "Wpedantic" unit)
    <> constructor "Wno" (CFlag_Wno <$> auto)
    <> constructor "Define" (record (CFlag_Define <$> field "name" auto <*> field "value" auto))
    <> constructor "Undef" (CFlag_Undef <$> auto)
    <> constructor "Include" (CFlag_Include <$> auto)
    <> constructor "System" (CFlag_System <$> auto)
    <> (CFlag_PIC <$ constructor "PIC" unit)
    <> (CFlag_PIE <$ constructor "PIE" unit)
    <> (CFlag_Static <$ constructor "Static" unit)
    <> (CFlag_Shared <$ constructor "Shared" unit)
    <> constructor "March" (CFlag_March <$> auto)
    <> constructor "Mtune" (CFlag_Mtune <$> auto)
    <> (CFlag_Native <$ constructor "Native" unit)
    <> constructor "Debug" (CFlag_Debug <$> autoWith opts)
    <> constructor "Sanitize" (CFlag_Sanitize <$> autoWith opts)
    <> (CFlag_FunctionSections <$ constructor "FunctionSections" unit)
    <> (CFlag_DataSections <$ constructor "DataSections" unit)
    <> (CFlag_NoExceptions <$ constructor "NoExceptions" unit)
    <> (CFlag_NoRTTI <$ constructor "NoRTTI" unit)
    <> (CFlag_Pthread <$ constructor "Pthread" unit)
    <> constructor "Raw" (CFlag_Raw <$> auto)
    )

-- | Linker flags (matches LDFlags.dhall LDFlag union)
data LDFlag
  = LDFlag_Static
  | LDFlag_Shared
  | LDFlag_Pie
  | LDFlag_NoPie
  | LDFlag_Relocatable
  | LDFlag_Lib Text
  | LDFlag_LibPath Text
  | LDFlag_Rpath Text
  | LDFlag_RpathLink Text
  | LDFlag_Strip
  | LDFlag_StripDebug
  | LDFlag_ExportDynamic
  | LDFlag_AsNeeded
  | LDFlag_NoAsNeeded
  | LDFlag_GcSections
  | LDFlag_NoGcSections
  | LDFlag_PrintGcSections
  | LDFlag_Soname Text
  | LDFlag_VersionScript Text
  | LDFlag_LTOJobs Natural
  | LDFlag_Raw Text
  deriving stock (Show, Eq, Generic)

instance FromDhall LDFlag where
  autoWith _ = union
    ( (LDFlag_Static <$ constructor "Static" unit)
    <> (LDFlag_Shared <$ constructor "Shared" unit)
    <> (LDFlag_Pie <$ constructor "Pie" unit)
    <> (LDFlag_NoPie <$ constructor "NoPie" unit)
    <> (LDFlag_Relocatable <$ constructor "Relocatable" unit)
    <> constructor "Lib" (LDFlag_Lib <$> auto)
    <> constructor "LibPath" (LDFlag_LibPath <$> auto)
    <> constructor "Rpath" (LDFlag_Rpath <$> auto)
    <> constructor "RpathLink" (LDFlag_RpathLink <$> auto)
    <> (LDFlag_Strip <$ constructor "Strip" unit)
    <> (LDFlag_StripDebug <$ constructor "StripDebug" unit)
    <> (LDFlag_ExportDynamic <$ constructor "ExportDynamic" unit)
    <> (LDFlag_AsNeeded <$ constructor "AsNeeded" unit)
    <> (LDFlag_NoAsNeeded <$ constructor "NoAsNeeded" unit)
    <> (LDFlag_GcSections <$ constructor "GcSections" unit)
    <> (LDFlag_NoGcSections <$ constructor "NoGcSections" unit)
    <> (LDFlag_PrintGcSections <$ constructor "PrintGcSections" unit)
    <> constructor "Soname" (LDFlag_Soname <$> auto)
    <> constructor "VersionScript" (LDFlag_VersionScript <$> auto)
    <> constructor "LTOJobs" (LDFlag_LTOJobs <$> auto)
    <> constructor "Raw" (LDFlag_Raw <$> auto)
    )

-- -----------------------------------------------------------------------------
-- Toolchain (mirror src/armitage/dhall/Toolchain.dhall)
-- -----------------------------------------------------------------------------

data Toolchain = Toolchain
  { compiler :: Compiler
  , host :: Triple
  , target :: Triple
  , cflags :: [CFlag]
  , cxxflags :: [CFlag]
  , ldflags :: [LDFlag]
  , sysroot :: Maybe Text
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (FromDhall)

-- -----------------------------------------------------------------------------
-- Resource Types (mirror src/armitage/dhall/Resource.dhall)
-- -----------------------------------------------------------------------------

data Resource
  = Resource_Pure
  | Resource_Network
  | Resource_Auth Text
  | Resource_Sandbox Text
  | Resource_Filesystem Text
  deriving stock (Show, Eq, Generic)

instance FromDhall Resource where
  autoWith _ = union
    ( (Resource_Pure <$ constructor "Pure" unit)
    <> (Resource_Network <$ constructor "Network" unit)
    <> constructor "Auth" (Resource_Auth <$> auto)
    <> constructor "Sandbox" (Resource_Sandbox <$> auto)
    <> constructor "Filesystem" (Resource_Filesystem <$> auto)
    )

-- -----------------------------------------------------------------------------
-- Build Target Types (mirror src/armitage/dhall/Build.dhall)
-- -----------------------------------------------------------------------------

data Dep
  = Dep_Local Text
  | Dep_External { hash :: Text, dname :: Text }
  | Dep_PkgConfig Text
  | Dep_Flake Text              -- "nixpkgs#openssl", ".#libfoo"
  deriving stock (Show, Eq, Generic)

instance FromDhall Dep where
  autoWith _ = union
    ( constructor "Local" (Dep_Local <$> auto)
    <> constructor "External" (record (Dep_External <$> field "hash" auto <*> field "name" auto))
    <> constructor "PkgConfig" (Dep_PkgConfig <$> auto)
    <> constructor "Flake" (Dep_Flake <$> auto)
    )

data Src
  = Src_Files [Text]
  | Src_Fetch { url :: Text, fhash :: Text }
  | Src_Git { url :: Text, rev :: Text, ghash :: Text }
  deriving stock (Show, Eq, Generic)

instance FromDhall Src where
  autoWith _ = union
    ( constructor "Files" (Src_Files <$> auto)
    <> constructor "Fetch" (record (Src_Fetch <$> field "url" auto <*> field "hash" auto))
    <> constructor "Git" (record (Src_Git <$> field "url" auto <*> field "rev" auto <*> field "hash" auto))
    )

data Target = Target
  { targetName :: Text
  , srcs :: Src
  , deps :: [Dep]
  , toolchain :: Toolchain
  , requires :: [Resource]
  }
  deriving stock (Show, Eq, Generic)

instance FromDhall Target where
  autoWith _ = record
    ( Target
      <$> field "name" auto
      <*> field "srcs" auto
      <*> field "deps" auto
      <*> field "toolchain" auto
      <*> field "requires" auto
    )

-- -----------------------------------------------------------------------------
-- Loading from Dhall
-- -----------------------------------------------------------------------------

-- | Load a single target from a Dhall file
loadTarget :: FilePath -> IO Target
loadTarget = inputFile auto

-- | Load multiple targets from a Dhall file that returns a list
loadTargets :: FilePath -> IO [Target]
loadTargets = inputFile auto

-- | Load a toolchain definition
loadToolchain :: FilePath -> IO Toolchain
loadToolchain = inputFile auto

-- -----------------------------------------------------------------------------
-- Conversion to Builder Types
-- -----------------------------------------------------------------------------

-- | Convert Resource to Builder.Coeffect
resourceToCoeffect :: Resource -> Builder.Coeffect
resourceToCoeffect = \case
  Resource_Pure -> Builder.Pure
  Resource_Network -> Builder.Network
  Resource_Auth provider -> Builder.Auth provider
  Resource_Sandbox n -> Builder.Sandbox n
  Resource_Filesystem path -> Builder.Filesystem (T.unpack path)

-- | Convert a Target to a Derivation
--
-- This is where Dhall configuration becomes executable.
targetToDerivation :: Target -> Builder.Derivation
targetToDerivation Target {..} =
  Builder.Derivation
    { Builder.drvName = targetName
    , Builder.drvSystem = renderTriple (target toolchain)
    , Builder.drvBuilder = "/bin/sh"  -- Would resolve from toolchain
    , Builder.drvArgs = ["-c", buildCommand]
    , Builder.drvEnv = buildEnv
    , Builder.drvInputDrvs = Map.empty  -- Would resolve deps
    , Builder.drvInputSrcs = sourcePaths
    , Builder.drvOutputs = Map.singleton "out" Builder.DrvOutput
        { Builder.doPath = Nothing  -- CA derivation
        , Builder.doHashAlgo = Nothing
        , Builder.doHash = Nothing
        }
    , Builder.drvContentAddressed = True
    }
  where
    buildCommand = "echo 'TODO: generate from toolchain'"

    buildEnv = Map.fromList
      [ ("CC", compilerPath (compiler toolchain))
      , ("CXX", compilerPath (compiler toolchain))
      , ("CFLAGS", T.unwords $ map renderCFlag (cflags toolchain))
      , ("CXXFLAGS", T.unwords $ map renderCFlag (cxxflags toolchain))
      , ("LDFLAGS", T.unwords $ map renderLDFlag (ldflags toolchain))
      ]

    sourcePaths = case srcs of
      Src_Files fs -> map T.unpack fs
      Src_Fetch {} -> []  -- Would be fetched
      Src_Git {} -> []    -- Would be cloned

    compilerPath = \case
      Compiler_Clang {} -> "clang"
      Compiler_NVClang {} -> "nv-clang"
      Compiler_GCC {} -> "gcc"
      Compiler_NVCC {} -> "nvcc"
      Compiler_Rustc {} -> "rustc"
      Compiler_GHC {} -> "ghc"
      Compiler_Lean {} -> "lean"

-- -----------------------------------------------------------------------------
-- Rendering
-- -----------------------------------------------------------------------------

-- | Render Triple to LLVM triple string
renderTriple :: Triple -> Text
renderTriple Triple {..} =
  T.intercalate "-" $ filter (not . T.null)
    [ renderArch arch
    , renderVendor vendor
    , renderOS os
    , renderABI abi
    ]

renderArch :: Arch -> Text
renderArch = \case
  Arch_x86_64 -> "x86_64"
  Arch_aarch64 -> "aarch64"
  Arch_riscv64 -> "riscv64"
  Arch_wasm32 -> "wasm32"
  Arch_armv7 -> "armv7"

renderVendor :: Vendor -> Text
renderVendor = \case
  Vendor_unknown -> "unknown"
  Vendor_pc -> "pc"
  Vendor_apple -> "apple"
  Vendor_nvidia -> "nvidia"

renderOS :: OS -> Text
renderOS = \case
  OS_linux -> "linux"
  OS_darwin -> "darwin"
  OS_windows -> "windows"
  OS_wasi -> "wasi"
  OS_none -> "none"

renderABI :: ABI -> Text
renderABI = \case
  ABI_gnu -> "gnu"
  ABI_musl -> "musl"
  ABI_eabi -> "eabi"
  ABI_eabihf -> "eabihf"
  ABI_msvc -> "msvc"
  ABI_none -> ""

-- | Render CPU for -march flag
renderCpu :: Cpu -> Text
renderCpu = \case
  Cpu_generic -> "generic"
  Cpu_native -> "native"
  Cpu_x86_64_v2 -> "x86-64-v2"
  Cpu_x86_64_v3 -> "x86-64-v3"
  Cpu_x86_64_v4 -> "x86-64-v4"
  Cpu_znver3 -> "znver3"
  Cpu_znver4 -> "znver4"
  Cpu_znver5 -> "znver5"
  Cpu_sapphirerapids -> "sapphirerapids"
  Cpu_alderlake -> "alderlake"
  Cpu_neoverse_v2 -> "neoverse-v2"
  Cpu_neoverse_n2 -> "neoverse-n2"
  Cpu_cortex_a78ae -> "cortex-a78ae"
  Cpu_cortex_a78c -> "cortex-a78c"
  Cpu_apple_m1 -> "apple-m1"
  Cpu_apple_m2 -> "apple-m2"
  Cpu_apple_m3 -> "apple-m3"
  Cpu_apple_m4 -> "apple-m4"

-- | Render GPU for CUDA -arch flag
renderGpu :: Gpu -> Maybe Text
renderGpu = \case
  Gpu_none -> Nothing
  Gpu_sm_80 -> Just "sm_80"
  Gpu_sm_86 -> Just "sm_86"
  Gpu_sm_87 -> Just "sm_87"
  Gpu_sm_89 -> Just "sm_89"
  Gpu_sm_90 -> Just "sm_90"
  Gpu_sm_90a -> Just "sm_90a"
  Gpu_sm_100 -> Just "sm_100"
  Gpu_sm_100a -> Just "sm_100a"
  Gpu_sm_120 -> Just "sm_120"

-- | Render CFlag to string
renderCFlag :: CFlag -> Text
renderCFlag = \case
  CFlag_Opt O0 -> "-O0"
  CFlag_Opt O1 -> "-O1"
  CFlag_Opt O2 -> "-O2"
  CFlag_Opt O3 -> "-O3"
  CFlag_Opt Os -> "-Os"
  CFlag_Opt Oz -> "-Oz"
  CFlag_Opt Og -> "-Og"
  CFlag_LTO LTO_off -> ""
  CFlag_LTO LTO_thin -> "-flto=thin"
  CFlag_LTO LTO_full -> "-flto"
  CFlag_StdC C89 -> "-std=c89"
  CFlag_StdC C99 -> "-std=c99"
  CFlag_StdC C11 -> "-std=c11"
  CFlag_StdC C17 -> "-std=c17"
  CFlag_StdC C23 -> "-std=c23"
  CFlag_StdCxx Cxx11 -> "-std=c++11"
  CFlag_StdCxx Cxx14 -> "-std=c++14"
  CFlag_StdCxx Cxx17 -> "-std=c++17"
  CFlag_StdCxx Cxx20 -> "-std=c++20"
  CFlag_StdCxx Cxx23 -> "-std=c++23"
  CFlag_Wall -> "-Wall"
  CFlag_Wextra -> "-Wextra"
  CFlag_Werror -> "-Werror"
  CFlag_Wpedantic -> "-Wpedantic"
  CFlag_Wno w -> "-Wno-" <> w
  CFlag_Define n Nothing -> "-D" <> n
  CFlag_Define n (Just v) -> "-D" <> n <> "=" <> v
  CFlag_Undef u -> "-U" <> u
  CFlag_Include p -> "-I" <> p
  CFlag_System p -> "-isystem " <> p
  CFlag_PIC -> "-fPIC"
  CFlag_PIE -> "-fPIE"
  CFlag_Static -> "-static"
  CFlag_Shared -> "-shared"
  CFlag_March a -> "-march=" <> a
  CFlag_Mtune t -> "-mtune=" <> t
  CFlag_Native -> "-march=native"
  CFlag_Debug G0 -> "-g0"
  CFlag_Debug G1 -> "-g1"
  CFlag_Debug G2 -> "-g2"
  CFlag_Debug G3 -> "-g3"
  CFlag_Sanitize San_address -> "-fsanitize=address"
  CFlag_Sanitize San_memory -> "-fsanitize=memory"
  CFlag_Sanitize San_thread -> "-fsanitize=thread"
  CFlag_Sanitize San_undefined -> "-fsanitize=undefined"
  CFlag_Sanitize San_leak -> "-fsanitize=leak"
  CFlag_FunctionSections -> "-ffunction-sections"
  CFlag_DataSections -> "-fdata-sections"
  CFlag_NoExceptions -> "-fno-exceptions"
  CFlag_NoRTTI -> "-fno-rtti"
  CFlag_Pthread -> "-pthread"
  CFlag_Raw r -> r

-- | Render LDFlag to string
renderLDFlag :: LDFlag -> Text
renderLDFlag = \case
  LDFlag_Static -> "-static"
  LDFlag_Shared -> "-shared"
  LDFlag_Pie -> "-pie"
  LDFlag_NoPie -> "-no-pie"
  LDFlag_Relocatable -> "-r"
  LDFlag_Lib l -> "-l" <> l
  LDFlag_LibPath p -> "-L" <> p
  LDFlag_Rpath p -> "-Wl,-rpath," <> p
  LDFlag_RpathLink p -> "-Wl,-rpath-link," <> p
  LDFlag_Strip -> "-s"
  LDFlag_StripDebug -> "-S"
  LDFlag_ExportDynamic -> "-Wl,--export-dynamic"
  LDFlag_AsNeeded -> "-Wl,--as-needed"
  LDFlag_NoAsNeeded -> "-Wl,--no-as-needed"
  LDFlag_GcSections -> "-Wl,--gc-sections"
  LDFlag_NoGcSections -> "-Wl,--no-gc-sections"
  LDFlag_PrintGcSections -> "-Wl,--print-gc-sections"
  LDFlag_Soname n -> "-Wl,-soname," <> n
  LDFlag_VersionScript p -> "-Wl,--version-script," <> p
  LDFlag_LTOJobs n -> "-flto-jobs=" <> T.pack (show n)
  LDFlag_Raw r -> r

-- (unit is imported from Dhall)
