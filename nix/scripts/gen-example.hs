{-# LANGUAGE OverloadedStrings #-}

-- | Example: Generate typed wrapper for ripgrep
module Main where

import Aleph.Script.Clap
import qualified Data.Text.IO as TIO
import System.IO (hPutStrLn, stderr)

main :: IO ()
main = do
    -- Read ripgrep short help (more manageable size)
    help <- TIO.readFile "corpus/rg-short.txt"
    let parsed = parseHelp help

    hPutStrLn stderr $ "Parsed " ++ show (sum $ map (length . secOptions) (helpSections parsed)) ++ " options"
    hPutStrLn stderr $ "Parsed " ++ show (length $ helpPositionals parsed) ++ " positional args"

    -- Generate the module
    TIO.putStrLn $ generateModule "Rg" "rg" parsed
