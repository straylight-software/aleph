{-# LANGUAGE OverloadedStrings #-}

-- | Quick test of Aleph.Script
module Main where

import Aleph.Script

main :: IO ()
main = script $ do
    echo "=== Aleph.Script Test ==="

    -- Test formatting (simple, single-arg formats)
    echo $ "Hello, " <> format s "World" <> "!"
    echo $ "Pi is approximately " <> format f 3.14159
    echo $ "Hex 255 = " <> format x 255

    -- Test file operations
    cwd <- pwd
    echo $ "Current directory: " <> format fp cwd

    files <- ls "."
    echo $ "Files in current dir: " <> format d (Prelude.length files)

    -- Test environment
    homeDir <- getEnvDefault "HOME" "/tmp"
    echo $ "Home: " <> homeDir

    -- Test timed operation
    (_, duration) <- timed $ sleep 0.1
    echo $ "Sleep took " <> format f duration <> " seconds"

    -- Test error handling
    _ <- errExit False $ run "false" []
    code <- exitCode
    echo $ "'false' exited with code " <> format d code

    echo "=== All tests passed! ==="
