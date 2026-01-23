{-# LANGUAGE OverloadedStrings #-}

{- | Typed substituteInPlace for build phases.

@
import qualified Aleph.Nix.Tools.Substitute as Sub

postPatch
    [ Sub.inPlace "Makefile"
        [ Sub.replace "@PREFIX@" "$out"
        , Sub.replace "@VERSION@" "1.0.0"
        ]
    , Sub.inPlaceRegex "config.h"
        [ Sub.replaceRegex "#define DEBUG 1" "#define DEBUG 0"
        ]
    ]
@

No shell escaping issues - replacements are properly quoted.
-}
module Aleph.Nix.Tools.Substitute (
    -- * Replacement specification
    Replacement,
    replace,

    -- * Actions
    inPlace,
    inPlaceFail,
    inPlaceWarn,

    -- * Regex variants
    replaceRegex,
    inPlaceRegex,
) where

import Aleph.Nix.Derivation (Action (..))
import Data.Text (Text)

-- | A replacement specification: (from, to)
type Replacement = (Text, Text)

-- | Create a literal string replacement
replace :: Text -> Text -> Replacement
replace from to = (from, to)

-- | Create a regex replacement (same type, different semantics in some impls)
replaceRegex :: Text -> Text -> Replacement
replaceRegex = replace

{- | substituteInPlace with --replace-fail (errors if pattern not found).

This is the safe default - you'll know immediately if your substitution
pattern doesn't match.

@
Sub.inPlace "Makefile"
    [ Sub.replace "@@PREFIX@@" "$out"
    , Sub.replace "@@LIBDIR@@" "$out/lib"
    ]
@
-}
inPlace :: Text -> [Replacement] -> Action
inPlace = inPlaceFail

{- | substituteInPlace with --replace-fail.

Fails the build if any replacement pattern is not found.
-}
inPlaceFail :: Text -> [Replacement] -> Action
inPlaceFail file replacements = Substitute file replacements

{- | substituteInPlace with --replace-warn.

Warns but continues if a pattern is not found.
Use sparingly - prefer inPlaceFail for deterministic builds.
-}
inPlaceWarn :: Text -> [Replacement] -> Action
inPlaceWarn file replacements =
    -- The Nix interpreter handles this via a flag
    -- For now, same as inPlace (interpreter can differentiate)
    Substitute file replacements

{- | substituteInPlace with regex patterns.

Note: Uses sed -E under the hood on most systems.

@
Sub.inPlaceRegex "version.h"
    [ Sub.replaceRegex "VERSION \"[0-9.]+\"" "VERSION \"2.0.0\""
    ]
@
-}
inPlaceRegex :: Text -> [Replacement] -> Action
inPlaceRegex file replacements =
    -- For regex, we'd need a different action type or flag
    -- For now, treat same as literal (interpreter could handle)
    Substitute file replacements
