{-# LANGUAGE OverloadedStrings #-}

{- |
Run a command in a namespace with GPU device access.

Usage: gpu-run COMMAND [ARGS...]

Like fhs-run but with:
- Full /dev access (including GPU devices)
- /sys mounted (for GPU info)
- /run/current-system for nvidia drivers
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
            echoErr "Usage: gpu-run COMMAND [ARGS...]"
            echoErr ""
            echoErr "Run COMMAND in a namespace with GPU device access."
            echoErr ""
            echoErr "Examples:"
            echoErr "  gpu-run nvidia-smi"
            echoErr "  gpu-run python3 -c 'import torch; print(torch.cuda.is_available())'"
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
                        & Bwrap.roBind "/sys" "/sys"
                        & Bwrap.tmpfs "/tmp"
                        -- NixOS specifics for GPU
                        & Bwrap.roBind "/run/current-system" "/run/current-system"
                        -- Network/SSL
                        & Bwrap.roBind "/etc/resolv.conf" "/etc/resolv.conf"
                        & Bwrap.roBind "/etc/ssl" "/etc/ssl"
                        -- User directories (read-write)
                        & Bwrap.bind (unpack homeDir) (unpack homeDir)
                        & Bwrap.bind cwd cwd
                        & Bwrap.chdir cwd
                        -- Environment
                        & Bwrap.setenv "PATH" "/run/current-system/sw/bin:/nix/store/bin:/usr/local/bin:/usr/bin:/bin"
                        & Bwrap.setenv "HOME" homeDir
                        & Bwrap.dieWithParent

            Bwrap.exec sandbox (map pack cmd)
