{-# LANGUAGE OverloadedStrings #-}

{- | Typed patchelf actions for build phases.

@
import qualified Aleph.Nix.Tools.PatchElf as PatchElf

postFixup
    [ PatchElf.setRpath "bin/myapp"
        [ PatchElf.rpathOut "lib"
        , PatchElf.rpathPkg "openssl" "lib"
        ]
    , PatchElf.setInterpreter "bin/myapp" PatchElf.Interpreter
    ]
@

The patchelf package is automatically added to nativeBuildInputs.
-}
module Aleph.Nix.Tools.PatchElf (
    -- * Rpath manipulation
    setRpath,
    addRpath,

    -- * Rpath entries (typed)
    RpathEntry (..),
    rpathOut,
    rpathPkg,
    rpathOrigin,

    -- * Interpreter
    setInterpreter,

    -- * Other operations
    shrinkRpath,
    printRpath,
) where

import Aleph.Nix.Derivation (Action (..))
import Aleph.Nix.Types (PkgRef (..), RpathEntry (..), rpathOut, rpathPkg)
import Data.Text (Text)
import qualified Data.Text as T

-- | The patchelf package reference
patchelfPkg :: PkgRef
patchelfPkg = PkgRef "patchelf"

-- | \$ORIGIN-relative rpath (for relocatable binaries)
rpathOrigin :: Text -> RpathEntry
rpathOrigin suffix = RpathLit ("$ORIGIN" <> suffix)

-- | Convert RpathEntry to the string form for patchelf
rpathToText :: RpathEntry -> Text
rpathToText (RpathOut path) = "$out/" <> path
rpathToText (RpathPkg pkg subpath) = "${" <> pkg <> "}/" <> subpath
rpathToText (RpathLit lit) = lit

{- | Set the rpath of an ELF binary.

@
PatchElf.setRpath "bin/myapp"
    [ PatchElf.rpathOut "lib"           -- $out/lib
    , PatchElf.rpathPkg "zlib" "lib"    -- ${zlib}/lib
    , PatchElf.rpathOrigin "/../lib"    -- $ORIGIN/../lib
    ]
@
-}
setRpath :: Text -> [RpathEntry] -> Action
setRpath binary entries =
    ToolRun
        patchelfPkg
        [ "--set-rpath"
        , T.intercalate ":" (map rpathToText entries)
        , binary
        ]

{- | Add to the rpath of an ELF binary.

@
PatchElf.addRpath "lib/libfoo.so" [PatchElf.rpathPkg "openssl" "lib"]
@
-}
addRpath :: Text -> [RpathEntry] -> Action
addRpath binary entries =
    ToolRun
        patchelfPkg
        [ "--add-rpath"
        , T.intercalate ":" (map rpathToText entries)
        , binary
        ]

{- | Set the interpreter (dynamic linker) of an ELF binary.

@
PatchElf.setInterpreter "bin/myapp" "${glibc}/lib/ld-linux-x86-64.so.2"
@
-}
setInterpreter :: Text -> Text -> Action
setInterpreter binary interpreter =
    ToolRun patchelfPkg ["--set-interpreter", interpreter, binary]

{- | Shrink the rpath by removing unused entries.

@
PatchElf.shrinkRpath "bin/myapp"
@
-}
shrinkRpath :: Text -> Action
shrinkRpath binary =
    ToolRun patchelfPkg ["--shrink-rpath", binary]

{- | Print the rpath of an ELF binary (for debugging).

@
PatchElf.printRpath "bin/myapp"
@
-}
printRpath :: Text -> Action
printRpath binary =
    ToolRun patchelfPkg ["--print-rpath", binary]
