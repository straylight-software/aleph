{-# LANGUAGE LambdaCase #-}

-- | Aleph.Build.Flags
-- Sum types matching Dhall exactly. No strings.
module Aleph.Build.Flags
  ( -- Build configuration
    BuildType (..)
  , Linkage (..)
  , Optimization (..)
  , Sanitizer (..)
  , LTO (..)
  , PIC (..)
  , SIMD (..)

    -- CMake flag generation
  , buildTypeFlag
  , linkageFlags
  , ltoFlags
  , picFlags
  , optimizationFlag
  , sanitizerFlags
  , simdFlags

    -- Autotools flag generation
  , linkageAutotools
  , picAutotools
  ) where

-- | CMake build type
data BuildType
  = Release
  | Debug
  | RelWithDebInfo
  | MinSizeRel
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Static vs shared linking
data Linkage
  = Static
  | Shared
  | Both
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Optimization level
data Optimization
  = O0
  | O1
  | O2
  | O3
  | Os
  | Oz
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Sanitizer selection
data Sanitizer
  = Address
  | Thread
  | Memory
  | UndefinedBehavior
  | NoSanitizer
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Link-time optimization
data LTO
  = LTOOff
  | LTOThin
  | LTOFull
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Position-independent code
data PIC
  = PICOn
  | PICOff
  | PICDefault
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | SIMD instruction set
data SIMD
  = SIMDNone
  | SSE2
  | SSE4
  | AVX
  | AVX2
  | AVX512
  | Neon
  deriving (Show, Eq, Ord, Enum, Bounded)

--------------------------------------------------------------------------------
-- CMake flag generation
--------------------------------------------------------------------------------

buildTypeFlag :: BuildType -> String
buildTypeFlag = \case
  Release -> "Release"
  Debug -> "Debug"
  RelWithDebInfo -> "RelWithDebInfo"
  MinSizeRel -> "MinSizeRel"

linkageFlags :: Linkage -> [String]
linkageFlags = \case
  Static -> ["-DBUILD_SHARED_LIBS=OFF"]
  Shared -> ["-DBUILD_SHARED_LIBS=ON"]
  Both -> []

ltoFlags :: LTO -> [String]
ltoFlags = \case
  LTOOff -> []
  LTOThin -> ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DLLVM_ENABLE_LTO=Thin"]
  LTOFull -> ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DLLVM_ENABLE_LTO=Full"]

picFlags :: PIC -> [String]
picFlags = \case
  PICOn -> ["-DCMAKE_POSITION_INDEPENDENT_CODE=ON"]
  PICOff -> ["-DCMAKE_POSITION_INDEPENDENT_CODE=OFF"]
  PICDefault -> []

optimizationFlag :: Optimization -> String
optimizationFlag = \case
  O0 -> "-O0"
  O1 -> "-O1"
  O2 -> "-O2"
  O3 -> "-O3"
  Os -> "-Os"
  Oz -> "-Oz"

sanitizerFlags :: Sanitizer -> [String]
sanitizerFlags = \case
  Address -> ["-fsanitize=address"]
  Thread -> ["-fsanitize=thread"]
  Memory -> ["-fsanitize=memory"]
  UndefinedBehavior -> ["-fsanitize=undefined"]
  NoSanitizer -> []

simdFlags :: SIMD -> [String]
simdFlags = \case
  SIMDNone -> []
  SSE2 -> ["-msse2"]
  SSE4 -> ["-msse4.1", "-msse4.2"]
  AVX -> ["-mavx"]
  AVX2 -> ["-mavx2"]
  AVX512 -> ["-mavx512f"]
  Neon -> ["-mfpu=neon"]

--------------------------------------------------------------------------------
-- Autotools flag generation
--------------------------------------------------------------------------------

linkageAutotools :: Linkage -> [String]
linkageAutotools = \case
  Static -> ["--disable-shared", "--enable-static"]
  Shared -> ["--enable-shared", "--disable-static"]
  Both -> ["--enable-shared", "--enable-static"]

picAutotools :: PIC -> [String]
picAutotools = \case
  PICOn -> ["--with-pic"]
  PICOff -> ["--without-pic"]
  PICDefault -> []
