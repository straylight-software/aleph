#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

-- \|
-- nix-ci: CI-optimized Nix invocation
--
-- Uses eval cache (good for reproducibility), enables verbose output.
-- See RFC-005 for rationale.
--
-- Usage:
--   nix-ci build .#foo
--   nix-ci flake check

import Aleph.Script
import System.Environment (getArgs)
import System.Posix.Process (executeFile)

main :: IO ()
main = do
    args <- getArgs

    -- CI: use cache, but verbose logging
    let opts = ["--show-trace", "--print-build-logs", "--log-format", "raw"]

    executeFile "nix" True (args ++ opts) Nothing
