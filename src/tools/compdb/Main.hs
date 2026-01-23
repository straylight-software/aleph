-- src/tools/compdb/Main.hs
--
-- compdb: Generate compile_commands.json for clangd
--
-- Runs buck2 bxl to get compilation database, expands argsfiles,
-- and writes a flattened compile_commands.json at repo root.

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (forM, when)
import Data.Aeson (FromJSON, ToJSON, eitherDecodeStrict, encode)
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.List (isPrefixOf)
import Data.Maybe (fromMaybe)
import GHC.Generics (Generic)
import Options.Applicative
import System.Directory (doesFileExist, getCurrentDirectory)
import System.Exit (ExitCode (..), exitFailure)
import System.Process (readProcessWithExitCode)
import Text.Read (readMaybe)

-- | A single compilation database entry (from buck2 bxl output)
data CompileCommand = CompileCommand
  { arguments :: [String]
  , directory :: FilePath
  , file :: FilePath
  }
  deriving (Show, Generic)

instance FromJSON CompileCommand
instance ToJSON CompileCommand

-- | CLI options
data Options = Options
  { optTargets :: [String]
  , optOutput :: FilePath
  , optVerbose :: Bool
  }
  deriving (Show)

optionsParser :: Parser Options
optionsParser =
  Options
    <$> many
      ( strArgument
          ( metavar "TARGETS"
              <> help "Buck2 targets (default: //...)"
          )
      )
    <*> strOption
      ( long "output"
          <> short 'o'
          <> metavar "FILE"
          <> value "compile_commands.json"
          <> help "Output file (default: compile_commands.json)"
      )
    <*> switch
      ( long "verbose"
          <> short 'v'
          <> help "Verbose output"
      )

main :: IO ()
main = do
  opts <- execParser parserInfo
  run opts
  where
    parserInfo =
      info
        (optionsParser <**> helper)
        ( fullDesc
            <> progDesc "Generate compile_commands.json for clangd"
            <> header "compdb - Buck2 compilation database generator"
        )

run :: Options -> IO ()
run Options {..} = do
  let targets = if null optTargets then ["//..."] else optTargets

  when optVerbose $
    putStrLn $ "Generating compdb for: " ++ unwords targets

  -- Run buck2 bxl
  let bxlArgs =
        ["bxl", "prelude//cxx/tools/compilation_database.bxl:generate", "--"]
          ++ concatMap (\t -> ["--targets", t]) targets

  when optVerbose $
    putStrLn $ "Running: buck2 " ++ unwords bxlArgs

  (exitCode, stdout, stderr) <- readProcessWithExitCode "buck2" bxlArgs ""

  case exitCode of
    ExitFailure code -> do
      putStrLn $ "buck2 bxl failed with code " ++ show code
      putStrLn stderr
      exitFailure
    ExitSuccess -> do
      -- Parse the output path from buck2
      let outputPath = head $ lines stdout
      when optVerbose $
        putStrLn $ "Reading: " ++ outputPath

      -- Read and parse the compilation database
      compdbBytes <- BS.readFile outputPath
      case eitherDecodeStrict compdbBytes of
        Left err -> do
          putStrLn $ "Failed to parse compilation database: " ++ err
          exitFailure
        Right commands -> do
          -- Expand argsfiles in each command
          expanded <- forM (commands :: [CompileCommand]) expandArgsfiles

          -- Write flattened output
          LBS.writeFile optOutput (encodePretty expanded)
          putStrLn $ "Wrote " ++ show (length expanded) ++ " entries to " ++ optOutput

-- | Expand @file references in arguments
expandArgsfiles :: CompileCommand -> IO CompileCommand
expandArgsfiles cmd = do
  expandedArgs <- expandArgs (arguments cmd)
  pure cmd {arguments = expandedArgs}

-- | Recursively expand @file arguments
expandArgs :: [String] -> IO [String]
expandArgs [] = pure []
expandArgs (arg : rest)
  | "@" `isPrefixOf` arg = do
      let filepath = drop 1 arg
      exists <- doesFileExist filepath
      if exists
        then do
          content <- readFile filepath
          let fileArgs = parseArgsFile content
          expanded <- expandArgs fileArgs
          restExpanded <- expandArgs rest
          pure $ expanded ++ restExpanded
        else do
          -- File doesn't exist, keep the @file reference
          restExpanded <- expandArgs rest
          pure $ arg : restExpanded
  | otherwise = do
      restExpanded <- expandArgs rest
      pure $ arg : restExpanded

-- | Parse an argsfile, handling quoted strings and @file references
parseArgsFile :: String -> [String]
parseArgsFile = concatMap parseLine . lines
  where
    parseLine line
      | null stripped = []
      | head stripped == '#' = [] -- comment
      | head stripped == '"' = [parseQuoted stripped]
      | otherwise = [stripped]
      where
        stripped = dropWhile (== ' ') line

    parseQuoted s =
      -- Simple quoted string handling: "foo bar" -> foo bar
      case readMaybe s of
        Just unquoted -> unquoted
        Nothing -> s
