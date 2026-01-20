{-# LANGUAGE OverloadedStrings #-}

{- |
Run a command in a minimal FHS-like namespace.

Usage: fhs-run COMMAND [ARGS...]

Creates a namespace with:
- /nix/store (read-only)
- /dev, /proc, /tmp
- Your home directory and current directory (read-write)
- Basic PATH including coreutils
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import Data.Function ((&))
import qualified Data.List as L
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> script $ do
            echoErr "Usage: fhs-run COMMAND [ARGS...]"
            echoErr ""
            echoErr "Run COMMAND in a minimal FHS namespace."
            echoErr ""
            echoErr "Examples:"
            echoErr "  fhs-run bash"
            echoErr "  fhs-run python3 script.py"
            exit 1
        cmd -> script $ do
            homeDir <- getEnvDefault "HOME" "/root"
            cwd <- pwd

            let sandbox =
                    Bwrap.defaults
                        -- Core system
                        & Bwrap.roBind "/nix/store" "/nix/store"
                        & Bwrap.devBind "/dev" "/dev"
                        & Bwrap.proc "/proc"
                        & Bwrap.tmpfs "/tmp"
                        & Bwrap.tmpfs "/run"
                        -- Network/SSL
                        & Bwrap.roBind "/etc/resolv.conf" "/etc/resolv.conf"
                        & Bwrap.roBind "/etc/ssl" "/etc/ssl"
                        -- User directories (read-write)
                        & Bwrap.bind (unpack homeDir) (unpack homeDir)
                        & Bwrap.bind cwd cwd
                        & Bwrap.chdir cwd
                        -- Environment
                        & Bwrap.setenv "PATH" "/nix/store/bin:/usr/local/bin:/usr/bin:/bin"
                        & Bwrap.setenv "HOME" homeDir
                        & Bwrap.dieWithParent

            Bwrap.exec sandbox (map pack cmd)
