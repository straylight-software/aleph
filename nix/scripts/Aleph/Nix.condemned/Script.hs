{-# LANGUAGE OverloadedStrings #-}

{- | Typed wrappers for compiled Aleph scripts

These functions wrap calls to compiled Haskell scripts from straylight.script.compiled.
Each script is invoked via the Tool action, providing type safety at the DrvSpec level.

@
import Aleph.Nix.DrvSpec
import qualified Aleph.Nix.Script as Script

pkg :: DrvSpec
pkg = defaultDrvSpec
    { phases = emptyPhases
        { fixup = [Script.verifyStaticOnly out]
        }
    }
@
-}
module Aleph.Nix.Script (
    -- * Build verification
    verifyStaticOnly,
) where

import Aleph.Nix.DrvSpec (Action (..), Expr (..), Ref)

-- ============================================================================
-- Build Verification Scripts
-- ============================================================================

{- | Verify a directory contains only static libraries

Checks that the directory tree contains no shared libraries (.so, .dylib, .dll).
Fails the build if any are found.

@
fixup = [Script.verifyStaticOnly out]
@
-}
verifyStaticOnly :: Ref -> Action
verifyStaticOnly dir =
    Tool
        { toolDep = "straylight.script.compiled.verify-static-only"
        , toolBin = "verify-static-only"
        , toolArgs = [ExprRef dir]
        }
