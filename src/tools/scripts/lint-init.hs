{-# LANGUAGE OverloadedStrings #-}

{- |
Install aleph-naught lint configs to current directory.

Usage: lint-init <configs-dir>

Copies lint configuration files for various tools:
- .clang-format, .clang-tidy (C/C++)
- ruff.toml (Python)
- biome.json (JS/TS)
- .stylua.toml (Lua)
- .rustfmt.toml (Rust)
- taplo.toml (TOML)

The configs-dir argument is the path to the aleph-naught configs directory,
typically substituted by Nix at build time.
-}
module Main where

import Aleph.Script hiding (FilePath)
import System.Environment (getArgs, getProgName)
import System.Exit (exitFailure)

-- Config files to copy: (source name in configs dir, destination name)
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
        [configsDir] -> runInstall configsDir
        _ -> do
            prog <- getProgName
            putStrLn $ "Usage: " ++ prog ++ " <configs-dir>"
            putStrLn "  configs-dir: path to aleph-naught lint configs directory"
            exitFailure

runInstall :: FilePath -> IO ()
runInstall configsDir = script $ do
    echo ":: Installing aleph-naught lint configs"

    mapM_ (installConfig configsDir) lintConfigs

    echo ":: Done - The Law has been established"

installConfig :: FilePath -> (String, String) -> Sh ()
installConfig configsDir (srcName, dstName) = do
    let src = configsDir </> srcName
    cp src dstName
    echo $ "  " <> pack dstName
