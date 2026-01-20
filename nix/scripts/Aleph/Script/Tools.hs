{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Aleph.Script.Tools
Description : Typed CLI tool wrappers for Aleph.Script

This module collects typed wrappers for common CLI tools.
Import tools qualified to avoid name conflicts:

@
import Aleph.Script
import qualified Aleph.Script.Tools.Rg as Rg
import qualified Aleph.Script.Tools.Fd as Fd
import qualified Aleph.Script.Tools.Bat as Bat

main = script $ do
  -- Search for TODOs
  matches <- Rg.rg Rg.defaults { Rg.ignoreCase = True } "TODO" ["."]

  -- Find Haskell files
  files <- Fd.findByExt "hs" ["."]

  -- Display a file with syntax highlighting
  Bat.bat_ Bat.defaults { Bat.language = Just "haskell" } ["src/Main.hs"]
@

== Available tools

=== Search & Find
  * "Aleph.Script.Tools.Rg" - ripgrep (fast regex search)
  * "Aleph.Script.Tools.Fd" - fd (fast file finder)

=== File Display
  * "Aleph.Script.Tools.Bat" - bat (cat with syntax highlighting)
  * "Aleph.Script.Tools.Delta" - delta (git diff viewer)

=== Code Quality
  * "Aleph.Script.Tools.Deadnix" - deadnix (find dead Nix code)
  * "Aleph.Script.Tools.Statix" - statix (Nix linter)
  * "Aleph.Script.Tools.Stylua" - stylua (Lua formatter)
  * "Aleph.Script.Tools.Taplo" - taplo (TOML toolkit)

=== Benchmarking & Stats
  * "Aleph.Script.Tools.Hyperfine" - hyperfine (command benchmarking)
  * "Aleph.Script.Tools.Tokei" - tokei (code statistics)
  * "Aleph.Script.Tools.Dust" - dust (disk usage)

=== Navigation
  * "Aleph.Script.Tools.Zoxide" - zoxide (smart cd)

=== Containers & JSON
  * "Aleph.Script.Tools.Jq" - jq (JSON processor)
  * "Aleph.Script.Tools.Crane" - crane (OCI image tool)
  * "Aleph.Script.Tools.Bwrap" - bwrap (bubblewrap sandbox)

=== GNU Coreutils & Classic Unix
  * "Aleph.Script.Tools.Ls" - ls (list directory contents)
  * "Aleph.Script.Tools.Grep" - grep (pattern matching)
  * "Aleph.Script.Tools.Sed" - sed (stream editor)
  * "Aleph.Script.Tools.Find" - find (file finder)
  * "Aleph.Script.Tools.Xargs" - xargs (build command lines)
  * "Aleph.Script.Tools.Tar" - tar (archive utility)
  * "Aleph.Script.Tools.Gzip" - gzip (compression)
  * "Aleph.Script.Tools.Wget" - wget (web downloader)
  * "Aleph.Script.Tools.Rsync" - rsync (remote sync)

== Adding new tools

New tool wrappers can be generated using the clap parser:

@
.\/gen-tool-wrapper.hs \<command\> > Weyl\/Script\/Tools\/\<Name\>.hs
@

Then review and adjust the generated code as needed.
-}
module Aleph.Script.Tools (
    -- * Re-exports for convenience

    -- | Note: Import qualified to avoid name conflicts between tools.

    -- ** Clap-based tools (Rust)
    module Rg,
    module Fd,
    module Bat,
    module Deadnix,
    module Delta,
    module Dust,
    module Hyperfine,
    module Statix,
    module Stylua,
    module Taplo,
    module Tokei,
    module Zoxide,

    -- ** GNU/Classic Unix tools
    module Ls,
    module Grep,
    module Sed,
    module Find,
    module Xargs,
    module Tar,
    module Gzip,
    module Wget,
    module Rsync,

    -- ** Container & JSON tools
    module Jq,
    module Crane,
    module Bwrap,
) where

-- Clap-based tools (Rust)

import qualified Aleph.Script.Tools.Bat as Bat
import qualified Aleph.Script.Tools.Deadnix as Deadnix
import qualified Aleph.Script.Tools.Delta as Delta
import qualified Aleph.Script.Tools.Dust as Dust
import qualified Aleph.Script.Tools.Fd as Fd
import qualified Aleph.Script.Tools.Hyperfine as Hyperfine
import qualified Aleph.Script.Tools.Rg as Rg
import qualified Aleph.Script.Tools.Statix as Statix
import qualified Aleph.Script.Tools.Stylua as Stylua
import qualified Aleph.Script.Tools.Taplo as Taplo
import qualified Aleph.Script.Tools.Tokei as Tokei
import qualified Aleph.Script.Tools.Zoxide as Zoxide

-- GNU/Classic Unix tools

import qualified Aleph.Script.Tools.Find as Find
import qualified Aleph.Script.Tools.Grep as Grep
import qualified Aleph.Script.Tools.Gzip as Gzip
import qualified Aleph.Script.Tools.Ls as Ls
import qualified Aleph.Script.Tools.Rsync as Rsync
import qualified Aleph.Script.Tools.Sed as Sed
import qualified Aleph.Script.Tools.Tar as Tar
import qualified Aleph.Script.Tools.Wget as Wget
import qualified Aleph.Script.Tools.Xargs as Xargs

-- Container & JSON tools

import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Aleph.Script.Tools.Crane as Crane
import qualified Aleph.Script.Tools.Jq as Jq
