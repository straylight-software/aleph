{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}

{- | Nix value types as seen from WASM modules.

This module provides:

1. Type tags for runtime Nix value inspection
2. Newtypes for type-safe Nix concepts (StorePath, System, etc.)

The newtypes ensure compile-time safety: you can't accidentally pass
a random Text where a StorePath is expected.
-}
module Aleph.Nix.Types (
    -- * Runtime type tags
    NixType (..),
    pattern TInt,
    pattern TFloat,
    pattern TBool,
    pattern TString,
    pattern TPath,
    pattern TNull,
    pattern TAttrs,
    pattern TList,
    pattern TFunction,
    fromTypeId,
    toTypeId,

    -- * Store paths
    StorePath (..),
    DrvPath (..),

    -- * System identification
    System (..),
    system_x86_64_linux,
    system_aarch64_linux,
    system_x86_64_darwin,
    system_aarch64_darwin,

    -- * Content hashes
    SriHash (..),
    sha256Hash,

    -- * Typed paths for build phases
    OutPath (..),
    SrcPath (..),
    PkgRef (..),
    RpathEntry (..),
    outPath,
    srcPath,
    pkgRef,
    rpathOut,
    rpathPkg,
    rpathLit,
) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Word (Word32)
import GHC.Generics (Generic)

-- | The type of a Nix value.
data NixType
    = -- | 64-bit signed integer
      NixInt
    | -- | 64-bit IEEE 754 float
      NixFloat
    | -- | Boolean
      NixBool
    | -- | UTF-8 string (may contain context)
      NixString
    | -- | Filesystem path
      NixPath
    | -- | The null value
      NixNull
    | -- | Attribute set (string-keyed map)
      NixAttrs
    | -- | Homogeneous list
      NixList
    | -- | Lambda or primop
      NixFunction
    | -- | Unknown type tag (for forward compat)
      NixUnknown Word32
    deriving (Eq, Show)

-- | Pattern synonyms for the type IDs as used in the host interface.
pattern TInt, TFloat, TBool, TString, TPath, TNull, TAttrs, TList, TFunction :: Word32
pattern TInt = 1
pattern TFloat = 2
pattern TBool = 3
pattern TString = 4
pattern TPath = 5
pattern TNull = 6
pattern TAttrs = 7
pattern TList = 8
pattern TFunction = 9

-- | Convert a type ID from the host to our ADT.
fromTypeId :: Word32 -> NixType
fromTypeId TInt = NixInt
fromTypeId TFloat = NixFloat
fromTypeId TBool = NixBool
fromTypeId TString = NixString
fromTypeId TPath = NixPath
fromTypeId TNull = NixNull
fromTypeId TAttrs = NixAttrs
fromTypeId TList = NixList
fromTypeId TFunction = NixFunction
fromTypeId n = NixUnknown n

-- | Convert our ADT back to a type ID.
toTypeId :: NixType -> Word32
toTypeId NixInt = TInt
toTypeId NixFloat = TFloat
toTypeId NixBool = TBool
toTypeId NixString = TString
toTypeId NixPath = TPath
toTypeId NixNull = TNull
toTypeId NixAttrs = TAttrs
toTypeId NixList = TList
toTypeId NixFunction = TFunction
toTypeId (NixUnknown n) = n

-- ============================================================================
-- Store paths
-- ============================================================================

{- | A Nix store path (e.g., "/nix/store/abc123-foo-1.0").

This is an *output* path - the result of building a derivation.
The hash in the path is either:

- Input-addressed: hash of the derivation closure (legacy, insane)
- Content-addressed: hash of the output contents (CA, sane)

With CA derivations always-on, you don't need to care which - the
store path is what you get, and it's determined by content.
-}
newtype StorePath = StorePath {unStorePath :: Text}
    deriving stock (Eq, Ord, Show, Generic)

{- | A derivation path (e.g., "/nix/store/xyz789-foo-1.0.drv").

This is the *recipe* for building something, not the result.
-}
newtype DrvPath = DrvPath {unDrvPath :: Text}
    deriving stock (Eq, Ord, Show, Generic)

-- ============================================================================
-- System identification
-- ============================================================================

{- | A Nix system identifier (e.g., "x86_64-linux").

This determines which builders can build the derivation and what
binaries are compatible.
-}
newtype System = System {unSystem :: Text}
    deriving stock (Eq, Ord, Show, Generic)

-- | x86_64-linux
system_x86_64_linux :: System
system_x86_64_linux = System "x86_64-linux"

-- | aarch64-linux
system_aarch64_linux :: System
system_aarch64_linux = System "aarch64-linux"

-- | x86_64-darwin (Intel Mac)
system_x86_64_darwin :: System
system_x86_64_darwin = System "x86_64-darwin"

-- | aarch64-darwin (Apple Silicon)
system_aarch64_darwin :: System
system_aarch64_darwin = System "aarch64-darwin"

-- ============================================================================
-- Content hashes
-- ============================================================================

{- | An SRI (Subresource Integrity) hash.

Format: "algorithm-base64hash" (e.g., "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB...")

This is the standard format for content hashes in Nix. The algorithm
prefix makes it self-describing and forward-compatible.
-}
newtype SriHash = SriHash {unSriHash :: Text}
    deriving stock (Eq, Ord, Show, Generic)

{- | Construct a SHA-256 SRI hash from a base64 hash value.

@
sha256Hash "Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
-- => SriHash "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
@
-}
sha256Hash :: Text -> SriHash
sha256Hash b64 = SriHash (T.concat ["sha256-", b64])

-- ============================================================================
-- Typed paths for build phases
-- ============================================================================

{- | A relative path within $out.

This is NOT a StorePath - it's a path fragment like "bin/foo" or "lib/libz.a".
Used in build phases to reference outputs being created.

Cannot be constructed from arbitrary Text - use 'outPath' smart constructor.
-}
newtype OutPath = OutPath {unOutPath :: Text}
    deriving stock (Eq, Ord, Show, Generic)

{- | A relative path within $src.

Used to reference files in the source tree during build.
-}
newtype SrcPath = SrcPath {unSrcPath :: Text}
    deriving stock (Eq, Ord, Show, Generic)

{- | A package reference - resolved to a store path at eval time.

This is how we reference dependencies without string interpolation.
@PkgRef "openssl"@ becomes @${pkgs.openssl}@ in Nix.
-}
newtype PkgRef = PkgRef {unPkgRef :: Text}
    deriving stock (Eq, Ord, Show, Generic)

{- | An rpath entry.

Can be:
- Relative to $out: @RpathOut "lib"@ → "$out/lib"
- A package reference: @RpathPkg "openssl" "lib"@ → "${openssl}/lib"
- Literal: @RpathLit "$ORIGIN/../lib"@ → "$ORIGIN/../lib"
-}
data RpathEntry
    = -- | Relative to $out
      RpathOut !Text
    | -- | Package name, subpath
      RpathPkg !Text !Text
    | -- | Literal string (e.g., $ORIGIN)
      RpathLit !Text
    deriving (Eq, Ord, Show, Generic)

-- | Smart constructors that prevent stringly-typed mistakes

-- | Path within $out (e.g., "bin/foo", "lib/libz.a")
outPath :: Text -> OutPath
outPath = OutPath

-- | Path within $src
srcPath :: Text -> SrcPath
srcPath = SrcPath

-- | Reference to a package by attribute name
pkgRef :: Text -> PkgRef
pkgRef = PkgRef

-- | Rpath relative to $out
rpathOut :: Text -> RpathEntry
rpathOut = RpathOut

-- | Rpath into a package's directory
rpathPkg :: Text -> Text -> RpathEntry
rpathPkg = RpathPkg

-- | Literal rpath (escape hatch for $ORIGIN etc)
rpathLit :: Text -> RpathEntry
rpathLit = RpathLit
