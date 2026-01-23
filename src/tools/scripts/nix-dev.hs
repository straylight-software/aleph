#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

-- \|
-- nix-dev: Development-optimized Nix invocation
--
-- Disables eval cache and enables verbose output.
-- See RFC-005 for rationale.
--
-- Usage:
--   nix-dev build .#foo
--   nix-dev develop

import Aleph.Script
import System.Environment (getArgs)
import System.Posix.Process (executeFile)

main :: IO ()
main = do
    args <- getArgs

    let globalOpts = ["--no-eval-cache", "--show-trace"]
        buildOpts = ["--print-build-logs", "--keep-failed"]

        extraOpts = case args of
            ("build" : _) -> globalOpts ++ buildOpts
            ("develop" : _) -> globalOpts ++ buildOpts
            ("run" : _) -> globalOpts ++ buildOpts
            ("shell" : _) -> globalOpts ++ buildOpts
            ("check" : _) -> globalOpts ++ buildOpts
            _ -> globalOpts

    -- exec nix with our args prepended
    executeFile "nix" True (args ++ extraOpts) Nothing
