{-# LANGUAGE OverloadedStrings #-}

{- |
Symlink aleph-naught lint configs to current directory.

Usage: lint-link <configs-dir>

Creates symlinks to lint configuration files for various tools:
- .clang-format, .clang-tidy (C/C++)
- ruff.toml (Python)
- biome.json (JS/TS)
- .stylua.toml (Lua)
- .rustfmt.toml (Rust)
- taplo.toml (TOML)

Unlike lint-init, this creates symlinks to the Nix store rather than
copying files. This means configs auto-update when aleph-naught is updated,
but requires the Nix store to be accessible.

The configs-dir argument is the path to the aleph-naught configs directory,
typically substituted by Nix at build time.
-}
module Main where

import Aleph.Script hiding (FilePath)
import Control.Exception (SomeException)
import qualified Control.Exception as E
import System.Directory (doesFileExist, doesPathExist)
import System.Environment (getArgs, getProgName)
import System.Exit (exitFailure)
import System.Posix.Files (createSymbolicLink, removeLink)

-- Config files to link: (source name in configs dir, destination name)
lintConfigs :: [(String, String)]
lintConfigs =
    [ (".clang-format", ".clang-format")
    , (".clang-tidy", ".clang-tidy")
    , ("ruff.toml", "ruff.toml")
    , ("biome.json", "biome.json")
    , (".stylua.toml", ".stylua.toml")
    , (".rustfmt.toml", ".rustfmt.toml")
    , ("taplo.toml", "taplo.toml")
    ]

main :: IO ()
main = do
    args <- getArgs
    case args of
        [configsDir] -> runLink configsDir
        _ -> do
            prog <- getProgName
            putStrLn $ "Usage: " ++ prog ++ " <configs-dir>"
            putStrLn "  configs-dir: path to aleph-naught lint configs directory"
            exitFailure

runLink :: FilePath -> IO ()
runLink configsDir = do
    putStrLn ":: Symlinking aleph-naught lint configs"

    mapM_ (linkConfig configsDir) lintConfigs

    putStrLn ":: Done - The Law has been linked"

linkConfig :: FilePath -> (String, String) -> IO ()
linkConfig configsDir (srcName, dstName) = do
    let src = configsDir </> srcName

    -- Remove existing file/symlink if present
    exists <- doesPathExist dstName
    if exists
        then removeLink dstName `E.catch` \(_ :: SomeException) -> return ()
        else return ()

    -- Create new symlink
    createSymbolicLink src dstName
    putStrLn $ "  " ++ dstName ++ " -> " ++ src
