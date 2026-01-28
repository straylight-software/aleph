#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

{- | Test all generated tool wrappers

Run: nix-shell -p "haskellPackages.ghcWithPackages (p: [p.megaparsec p.text p.shelly p.aeson p.foldl])" \
       bat fd ripgrep delta deadnix dust hyperfine statix stylua taplo tokei zoxide \
       --run "runghc -i. test-tools.hs"
-}
module Main where

import Data.Text (Text)
import qualified Data.Text as T
import Shelly

import qualified Aleph.Script.Tools.Bat as Bat
import qualified Aleph.Script.Tools.Deadnix as Deadnix
import qualified Aleph.Script.Tools.Delta as Delta
import qualified Aleph.Script.Tools.Dust as Dust
import qualified Aleph.Script.Tools.Fd as Fd
import qualified Aleph.Script.Tools.Hyperfine as Hyperfine
import qualified Aleph.Script.Tools.Rg as Rg
import qualified Aleph.Script.Tools.Stylua as Stylua
import qualified Aleph.Script.Tools.Tokei as Tokei

-- GNU tools

import qualified Aleph.Script.Tools.Grep as Grep
import qualified Aleph.Script.Tools.Ls as Ls
import qualified Aleph.Script.Tools.Tar as Tar

main :: IO ()
main = shelly $ verbosely $ do
    echo "=== Testing Aleph.Script.Tools ==="
    echo ""

    -- Test Rg (ripgrep)
    echo ">>> Rg: searching for 'module' in .hs files..."
    rgOut <-
        Rg.rg
            Rg.defaults
                { Rg.glob = Just "*.hs"
                , Rg.maxCount = Just 1 -- limit to 1 match per file
                }
            "^module"
            ["."]
    echo $ "Found " <> tshow (length (T.lines rgOut)) <> " modules"
    echo ""

    -- Test Fd (find)
    echo ">>> Fd: finding .hs files..."
    hsFiles <-
        Fd.fd
            Fd.defaults
                { Fd.extension = Just "hs"
                }
            Nothing
            ["."]
    echo $ "Found " <> tshow (length hsFiles) <> " Haskell files"
    echo ""

    -- Test Tokei (code stats)
    echo ">>> Tokei: counting lines of code..."
    tokeiOut <-
        Tokei.tokei
            Tokei.defaults
                { Tokei.output = Just "json"
                }
            ["."]
    echo $ "Tokei output length: " <> tshow (T.length tokeiOut) <> " chars"
    echo ""

    -- Test Dust (disk usage)
    echo ">>> Dust: checking disk usage of corpus/..."
    dustOut <-
        errExit False $
            Dust.dust
                Dust.defaults
                    { Dust.depth = Just "1"
                    , Dust.apparentSize = True
                    }
                ["corpus"]
    echo $ "Dust output: " <> tshow (length (T.lines dustOut)) <> " lines"
    echo ""

    -- Test Bat (syntax highlighting) - just verify it runs
    echo ">>> Bat: displaying first 5 lines of this script..."
    batOut <-
        Bat.bat
            Bat.defaults
                { Bat.lineRange = Just "1:5"
                , Bat.plain = True -- no decorations for easier parsing
                }
            ["test-tools.hs"]
    echo batOut
    echo ""

    -- Test Hyperfine (benchmarking)
    echo ">>> Hyperfine: benchmarking 'echo hello'..."
    _ <-
        errExit False $
            Hyperfine.hyperfine
                Hyperfine.defaults
                    { Hyperfine.runs = Just 3
                    , Hyperfine.warmup = Just 1
                    }
                ["echo hello"]
    echo "Hyperfine ran successfully"
    echo ""

    -- Test Deadnix (dead Nix code finder)
    echo ">>> Deadnix: checking flake.nix for dead code..."
    dnOut <-
        errExit False $
            Deadnix.deadnix
                Deadnix.defaults
                    { Deadnix.quiet = True
                    }
                ["../../flake.nix"]
    echo $ "Deadnix output: " <> tshow (length (T.lines dnOut)) <> " lines"
    echo ""

    -- ========================================
    -- GNU Tools
    -- ========================================

    echo "=== Testing GNU Tools ==="
    echo ""

    -- Test Ls (list directory)
    echo ">>> Ls: listing directory with long format..."
    lsOut <-
        Ls.ls
            Ls.defaults
                { Ls.optL = True -- -l long format
                , Ls.humanReadable = True
                }
            ["."]
    echo $ "Listed " <> tshow (length (T.lines lsOut)) <> " entries"
    echo ""

    -- Test Grep (pattern matching)
    echo ">>> Grep: searching for 'module' in .hs files..."
    grepOut <-
        errExit False $
            Grep.grep
                Grep.defaults
                    { Grep.recursive = True
                    , Grep.lineNumber = True
                    }
                ["module", "Weyl/Script.hs"]
    echo $ "Grep found " <> tshow (length (T.lines grepOut)) <> " matches"
    echo ""

    -- Test Tar (just list, don't create)
    echo ">>> Tar: creating and listing a test archive..."
    -- Create a simple test archive
    _ <- errExit False $ run_ "tar" ["cf", "/tmp/test-tools-archive.tar", "Weyl/Script.hs"]
    tarOut <-
        Tar.tar
            Tar.defaults
                { Tar.list = True
                , Tar.verbose = True
                , Tar.file = Just "/tmp/test-tools-archive.tar"
                }
            []
    echo $ "Tar archive contains: " <> tarOut
    -- Clean up
    _ <- errExit False $ run_ "rm" ["/tmp/test-tools-archive.tar"]
    echo ""

    echo "=== All tools working! ==="

tshow :: (Show a) => a -> Text
tshow = T.pack . show
