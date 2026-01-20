{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Test parser against real tool help outputs

Run: runghc -i. corpus-test.hs
-}
module Main where

import Aleph.Script.Clap
import Control.Monad (forM, forM_)
import Data.List (sortOn)
import Data.Maybe (isJust)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Directory (listDirectory)
import System.FilePath (takeBaseName, (</>))

-- ============================================================================
-- Statistics
-- ============================================================================

data ParseStats = ParseStats
    { statsTool :: String
    , statsVariant :: String
    , statsNumSections :: Int
    , statsNumOptions :: Int
    , statsNumPositionals :: Int
    , statsOptionsWithShort :: Int
    , statsOptionsWithLong :: Int
    , statsOptionsWithArg :: Int
    }
    deriving (Show)

computeStats :: String -> String -> ClapHelp -> ParseStats
computeStats tool variant ClapHelp{..} =
    let allOpts = concatMap secOptions helpSections
     in ParseStats
            { statsTool = tool
            , statsVariant = variant
            , statsNumSections = length helpSections
            , statsNumOptions = length allOpts
            , statsNumPositionals = length helpPositionals
            , statsOptionsWithShort = length $ filter (isJust . optShort) allOpts
            , statsOptionsWithLong = length $ filter (isJust . optLong) allOpts
            , statsOptionsWithArg = length $ filter (isJust . optArg) allOpts
            }

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = do
    putStrLn "=== Corpus Test: Real Tool Help Parsing ==="
    putStrLn ""

    let corpusDir = "corpus"
    files <- listDirectory corpusDir
    let txtFiles = sortOn id $ filter (\f -> ".txt" `isSuffixOf` f && not (null f)) files

    results <- forM txtFiles $ \file -> do
        let path = corpusDir </> file
            base = takeBaseName file
            (tool, variant) = case break (== '-') base of
                (t, '-' : v) -> (t, v)
                (t, _) -> (t, "unknown")

        content <- TIO.readFile path
        let help = parseHelp content
            stats = computeStats tool variant help
        return stats

    -- Print table header
    putStrLn $
        padRight 15 "Tool"
            ++ padRight 8 "Variant"
            ++ padRight 6 "Secs"
            ++ padRight 6 "Opts"
            ++ padRight 6 "Pos"
            ++ padRight 8 "Short"
            ++ padRight 8 "Long"
            ++ padRight 8 "Args"
    putStrLn $ replicate 70 '-'

    -- Print each result
    forM_ results $ \ParseStats{..} -> do
        putStrLn $
            padRight 15 statsTool
                ++ padRight 8 statsVariant
                ++ padRight 6 (show statsNumSections)
                ++ padRight 6 (show statsNumOptions)
                ++ padRight 6 (show statsNumPositionals)
                ++ padRight 8 (show statsOptionsWithShort)
                ++ padRight 8 (show statsOptionsWithLong)
                ++ padRight 8 (show statsOptionsWithArg)

    -- Summary
    let totalOpts = sum $ map statsNumOptions results
        totalPos = sum $ map statsNumPositionals results
    putStrLn $ replicate 70 '-'
    putStrLn $
        "Total: "
            ++ show (length results)
            ++ " files, "
            ++ show totalOpts
            ++ " options, "
            ++ show totalPos
            ++ " positionals parsed"
  where
    padRight n s = take n (s ++ repeat ' ')
    isSuffixOf suffix str = drop (length str - length suffix) str == suffix
