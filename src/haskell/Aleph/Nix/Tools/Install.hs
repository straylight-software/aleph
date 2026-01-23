{-# LANGUAGE OverloadedStrings #-}

{- | Typed install commands for build phases.

@
import qualified Aleph.Nix.Tools.Install as Install

installPhase
    [ Install.bin "src/myapp" "bin/myapp"
    , Install.lib "libfoo.so" "lib/libfoo.so"
    , Install.header "include/foo.h" "include/foo.h"
    , Install.data "share/icons/app.png" "share/icons/myapp/app.png"
    ]
@

Correct modes are applied automatically:
- bin: 0o755 (executable)
- lib: 0o644 (shared library)
- header: 0o644 (read-only)
- data: 0o644 (read-only)
-}
module Aleph.Nix.Tools.Install (
    -- * Common install patterns
    bin,
    lib,
    header,
    data_,
    doc,
    man,

    -- * Generic install with mode
    file,

    -- * Directory operations
    dir,
    tree,
) where

import Aleph.Nix.Derivation (Action (..))
import Data.Text (Text)

-- Standard Unix file modes
modeExecutable, modeReadOnly :: Int
modeExecutable = 0o755 -- rwxr-xr-x
modeReadOnly = 0o644 -- rw-r--r--

{- | Install an executable binary.

@
Install.bin "build/myapp" "bin/myapp"
@

Mode: 0o755 (rwxr-xr-x)
-}
bin :: Text -> Text -> Action
bin src dst = Install modeExecutable src dst

{- | Install a library (shared or static).

@
Install.lib "libfoo.so.1.0" "lib/libfoo.so.1.0"
@

Mode: 0o644 (rw-r--r--)
-}
lib :: Text -> Text -> Action
lib src dst = Install modeReadOnly src dst

{- | Install a header file.

@
Install.header "include/foo.h" "include/foo/foo.h"
@

Mode: 0o644 (rw-r--r--)
-}
header :: Text -> Text -> Action
header src dst = Install modeReadOnly src dst

{- | Install a data file.

@
Install.data_ "assets/icon.png" "share/myapp/icon.png"
@

Mode: 0o644 (rw-r--r--)

Note: Named data_ to avoid conflict with Prelude.
-}
data_ :: Text -> Text -> Action
data_ src dst = Install modeReadOnly src dst

{- | Install documentation.

@
Install.doc "README.md" "share/doc/myapp/README.md"
@

Mode: 0o644 (rw-r--r--)
-}
doc :: Text -> Text -> Action
doc src dst = Install modeReadOnly src dst

{- | Install a man page.

@
Install.man "man/myapp.1" "share/man/man1/myapp.1"
@

Mode: 0o644 (rw-r--r--)
-}
man :: Text -> Text -> Action
man src dst = Install modeReadOnly src dst

{- | Install a file with explicit mode.

@
Install.file 0o600 "secrets.conf" "etc/myapp/secrets.conf"
@
-}
file :: Int -> Text -> Text -> Action
file = Install

{- | Create a directory.

@
Install.dir "share/myapp/data"
@
-}
dir :: Text -> Action
dir = Mkdir

{- | Copy a directory tree.

@
Install.tree "doc/html" "share/doc/myapp/html"
@
-}
tree :: Text -> Text -> Action
tree = Copy
