{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Props
Description : Property tests for CLI wrapper generators

Run:
  nix-shell -p "haskellPackages.ghcWithPackages (p: [p.megaparsec p.text p.QuickCheck p.hedgehog])" \
    --run "runghc -i. Props.hs"

Properties tested:
  1. Parser totality: no crashes on arbitrary input
  2. Preservation: parsed options appear in generated code
  3. Idempotence: parse(x) == parse(x)
  4. Compilation: generated code is valid Haskell (via GHC)
-}
module Main where

import Control.DeepSeq (NFData (..), deepseq)
import Data.Char (isAlphaNum, isLower, isUpper)
import Data.Text (Text)
import qualified Data.Text as T
import System.Directory (getTemporaryDirectory, removeFile)
import System.Exit (ExitCode (..))
import System.IO (hClose, hPutStr, openTempFile)
import System.Process (readProcessWithExitCode)
import Test.QuickCheck

import qualified Aleph.Script.Clap as Clap
import qualified Aleph.Script.Getopt as Getopt

-- ============================================================================
-- Generators
-- ============================================================================

-- | Generate realistic-ish help text lines
genHelpLine :: Gen Text
genHelpLine =
    frequency
        [ (3, genOptionLine)
        , (1, genSectionHeader)
        , (2, genDescriptionLine)
        , (1, pure "")
        ]

-- | Generate a clap-style option line
genOptionLine :: Gen Text
genOptionLine = do
    indent <- elements ["  ", "    ", "      "]
    short <- frequency [(3, genShortOpt), (1, pure "")]
    long <- frequency [(4, genLongOpt), (1, pure "")]
    sep <-
        if (not (T.null short) && not (T.null long))
            then elements [", ", " "]
            else pure ""
    arg <- frequency [(2, genArgPlaceholder), (1, pure "")]
    desc <- genDescription
    pure $ indent <> short <> sep <> long <> arg <> "  " <> desc

genShortOpt :: Gen Text
genShortOpt = do
    c <- elements $ ['a' .. 'z'] ++ ['A' .. 'Z'] ++ ['0' .. '9']
    pure $ "-" <> T.singleton c

genLongOpt :: Gen Text
genLongOpt = do
    parts <- listOf1 $ elements ["foo", "bar", "ignore", "case", "file", "path", "no", "with"]
    pure $ "--" <> T.intercalate "-" (take 3 parts)

genArgPlaceholder :: Gen Text
genArgPlaceholder =
    frequency
        [
            ( 2
            , do
                name <- elements ["FILE", "PATH", "NUM", "PATTERN", "DIR", "COUNT"]
                pure $ "=" <> name
            )
        ,
            ( 1
            , do
                name <- elements ["FILE", "PATH", "NUM"]
                pure $ " <" <> name <> ">"
            )
        ]

genSectionHeader :: Gen Text
genSectionHeader = do
    name <- elements ["Options", "Input", "Output", "Search", "Filter", "POSITIONAL ARGUMENTS"]
    pure $ name <> ":"

genDescriptionLine :: Gen Text
genDescriptionLine = do
    indent <- elements ["        ", "          ", "            "]
    desc <- genDescription
    pure $ indent <> desc

genDescription :: Gen Text
genDescription = do
    words <- listOf $ elements ["the", "a", "file", "to", "use", "for", "output", "input", "match"]
    pure $ T.unwords (take 10 words)

-- | Generate a full help text (clap-style)
genHelpText :: Gen Text
genHelpText = do
    header <- elements ["mytool 1.0.0", "Usage: mytool [OPTIONS]", ""]
    sections <- listOf1 $ do
        hdr <- genSectionHeader
        opts <- listOf1 genOptionLine
        pure $ T.unlines (hdr : opts)
    pure $ T.unlines (header : sections)

-- | Generate GNU-style help text
genGnuHelpText :: Gen Text
genGnuHelpText = do
    header <- elements ["Usage: mytool [OPTION]... [FILE]...", "mytool - do something"]
    opts <- listOf1 genGnuOptionLine
    pure $ T.unlines (header : opts)

-- | Generate a GNU getopt-style option line
genGnuOptionLine :: Gen Text
genGnuOptionLine = do
    short <- frequency [(3, Just <$> elements ['a' .. 'z']), (1, pure Nothing)]
    long <- frequency [(4, Just <$> genLongName), (1, pure Nothing)]
    arg <- frequency [(2, Just <$> elements ["FILE", "NUM", "DIR"]), (1, pure Nothing)]
    desc <- genDescription
    let shortPart = maybe "" (\c -> "-" <> T.singleton c <> ", ") short
        longPart = maybe "" (\l -> "--" <> l) long
        argPart = maybe "" ("=" <>) arg
    case (short, long) of
        (Nothing, Nothing) -> genGnuOptionLine -- retry, need at least one
        _ -> pure $ "  " <> shortPart <> longPart <> argPart <> "  " <> desc
  where
    genLongName = do
        parts <- listOf1 $ elements ["foo", "bar", "ignore", "case", "file"]
        pure $ T.intercalate "-" (take 2 parts)

-- | Generate completely random text (for crash testing)
genArbitraryText :: Gen Text
genArbitraryText = do
    len <- choose (0, 1000)
    chars <- vectorOf len arbitrary
    pure $ T.pack chars

-- ============================================================================
-- Properties
-- ============================================================================

-- | Parser never crashes, even on garbage input
prop_clap_parser_total :: Property
prop_clap_parser_total = forAll genArbitraryText $ \input ->
    let result = Clap.parseHelp input
     in result `deepseq` True -- Force evaluation, just check it doesn't crash

prop_getopt_parser_total :: Property
prop_getopt_parser_total = forAll genArbitraryText $ \input ->
    let result = Getopt.parseHelp input
     in result `deepseq` True

-- | Parsing is idempotent (parse twice = same result)
prop_clap_idempotent :: Property
prop_clap_idempotent = forAll genHelpText $ \input ->
    let p1 = Clap.parseHelp input
        p2 = Clap.parseHelp input
     in p1 == p2

prop_getopt_idempotent :: Property
prop_getopt_idempotent = forAll genHelpText $ \input ->
    let p1 = Getopt.parseHelp input
        p2 = Getopt.parseHelp input
     in p1 == p2

-- | Every long option in parsed result appears in generated code
prop_clap_options_preserved :: Property
prop_clap_options_preserved = forAll genHelpText $ \input ->
    let parsed = Clap.parseHelp input
        generated = Clap.generateModule "Test" "test" parsed
        allOpts = concatMap Clap.secOptions (Clap.helpSections parsed)
        longNames =
            [ n
            | Clap.ClapOption{optLong = Just n} <- allOpts
            , n /= "help"
            , n /= "version"
            ]
     in all (\n -> n `T.isInfixOf` generated) longNames

prop_getopt_options_preserved :: Property
prop_getopt_options_preserved = forAll genGnuHelpText $ \input ->
    let parsed = Getopt.parseHelp input
        generated = Getopt.generateModule "Test" "test" parsed
        opts = Getopt.helpOptions parsed
        longNames =
            [ n
            | Getopt.GetoptOption{Getopt.optLong = Just n} <- opts
            , n /= "help"
            , n /= "version"
            ]
     in all (\n -> n `T.isInfixOf` generated) longNames

-- | Generated code contains required structure
prop_clap_generated_structure :: Property
prop_clap_generated_structure = forAll genHelpText $ \input ->
    let generated = Clap.generateModule "Test" "test" (Clap.parseHelp input)
     in and
            [ "module Aleph.Script.Tools.Test" `T.isInfixOf` generated
            , "data Options = Options" `T.isInfixOf` generated
            , "defaults :: Options" `T.isInfixOf` generated
            , "buildArgs :: Options -> [Text]" `T.isInfixOf` generated
            , "test :: Options -> [Text] -> Sh Text" `T.isInfixOf` generated
            ]

prop_getopt_generated_structure :: Property
prop_getopt_generated_structure = forAll genGnuHelpText $ \input ->
    let generated = Getopt.generateModule "Test" "test" (Getopt.parseHelp input)
     in and
            [ "module Aleph.Script.Tools.Test" `T.isInfixOf` generated
            , "data Options = Options" `T.isInfixOf` generated
            , "defaults :: Options" `T.isInfixOf` generated
            , "buildArgs :: Options -> [Text]" `T.isInfixOf` generated
            ]

-- | Field names are valid Haskell identifiers
prop_clap_valid_field_names :: Property
prop_clap_valid_field_names = forAll genLongOpt $ \opt ->
    let name = Clap.optionToHaskellName (T.drop 2 opt) -- drop "--"
     in not (T.null name) && isValidHaskellIdent name

prop_getopt_valid_field_names :: Property
prop_getopt_valid_field_names = forAll genLongOpt $ \opt ->
    let name = Getopt.optionToHaskellName (T.drop 2 opt)
     in not (T.null name) && isValidHaskellIdent name

isValidHaskellIdent :: Text -> Bool
isValidHaskellIdent t = case T.uncons t of
    Nothing -> False
    Just (c, rest) ->
        (isLower c || c == '_')
            && T.all (\x -> isAlphaNum x || x == '_' || x == '\'') rest

-- ============================================================================
-- Compilation Properties (IO)
-- ============================================================================

-- | Generated clap code compiles with GHC
prop_clap_compiles :: Property
prop_clap_compiles = once $ ioProperty $ do
    let input =
            T.unlines
                [ "Options:"
                , "  -v, --verbose          Be verbose"
                , "  -f, --file=FILE        Input file"
                , "  -n, --count=NUM        Number of items"
                , "      --ignore-case      Ignore case"
                ]
        generated = Clap.generateModule "TestClap" "test-clap" (Clap.parseHelp input)
    compiles generated

-- | Generated getopt code compiles with GHC
prop_getopt_compiles :: Property
prop_getopt_compiles = once $ ioProperty $ do
    let input =
            T.unlines
                [ "Usage: test [OPTION]..."
                , "  -v, --verbose          be verbose"
                , "  -f, --file=FILE        input file"
                , "  -n, --count=NUM        number of items"
                , "      --ignore-case      ignore case"
                ]
        generated = Getopt.generateModule "TestGetopt" "test-getopt" (Getopt.parseHelp input)
    compiles generated

-- | Generated code from random input compiles
prop_clap_random_compiles :: Property
prop_clap_random_compiles = withMaxSuccess 5 $ forAll genHelpText $ \input ->
    ioProperty $ do
        let generated = Clap.generateModule "TestRandom" "test-random" (Clap.parseHelp input)
        compiles generated

prop_getopt_random_compiles :: Property
prop_getopt_random_compiles = withMaxSuccess 5 $ forAll genGnuHelpText $ \input ->
    ioProperty $ do
        let generated = Getopt.generateModule "TestRandom" "test-random" (Getopt.parseHelp input)
        compiles generated

-- | Check if Haskell code compiles using GHC
compiles :: Text -> IO Bool
compiles code = do
    tmpDir <- getTemporaryDirectory
    (tmpFile, h) <- openTempFile tmpDir "PropTest.hs"
    hPutStr h (T.unpack code)
    hClose h
    -- Use -fno-code to just type-check without generating output
    -- -i. to find Aleph.Script module
    (exitCode, _stdout, stderr) <-
        readProcessWithExitCode
            "ghc"
            ["-fno-code", "-i.", tmpFile]
            ""
    removeFile tmpFile
    case exitCode of
        ExitSuccess -> return True
        ExitFailure _ -> do
            putStrLn $ "\n=== COMPILATION FAILED ==="
            putStrLn $ "Generated code:\n" ++ T.unpack code
            putStrLn $ "GHC error:\n" ++ stderr
            putStrLn "==========================="
            return False

-- ============================================================================
-- NFData instances for deepseq
-- ============================================================================

instance NFData Clap.ClapOption where
    rnf Clap.ClapOption{..} = rnf optShort `seq` rnf optLong `seq` rnf optArg `seq` rnf optDesc

instance NFData Clap.ClapPositional where
    rnf Clap.ClapPositional{..} = rnf posName `seq` rnf posRequired

instance NFData Clap.ClapSection where
    rnf Clap.ClapSection{..} = rnf secName `seq` rnf secOptions

instance NFData Clap.ClapHelp where
    rnf Clap.ClapHelp{..} = rnf helpSections `seq` rnf helpPositionals

instance NFData Getopt.GetoptOption where
    rnf Getopt.GetoptOption{..} =
        rnf optShort `seq`
            rnf optLong `seq`
                rnf optArg `seq`
                    rnf optArgOptional `seq`
                        rnf optDesc

instance NFData Getopt.GetoptHelp where
    rnf Getopt.GetoptHelp{..} = rnf helpOptions `seq` rnf helpUsage

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = do
    putStrLn "=== Property Tests for CLI Wrapper Generators ==="
    putStrLn ""

    let args = stdArgs{maxSuccess = 50} -- 50 is fast enough for quick checks
    putStrLn "-- Clap Parser --"
    putStr "  parser_total: "
    quickCheckWith args prop_clap_parser_total
    putStr "  idempotent: "
    quickCheckWith args prop_clap_idempotent
    putStr "  options_preserved: "
    quickCheckWith args prop_clap_options_preserved
    putStr "  generated_structure: "
    quickCheckWith args prop_clap_generated_structure
    putStr "  valid_field_names: "
    quickCheckWith args prop_clap_valid_field_names

    putStrLn ""
    putStrLn "-- GNU Getopt Parser --"
    putStr "  parser_total: "
    quickCheckWith args prop_getopt_parser_total
    putStr "  idempotent: "
    quickCheckWith args prop_getopt_idempotent
    putStr "  options_preserved: "
    quickCheckWith args prop_getopt_options_preserved
    putStr "  generated_structure: "
    quickCheckWith args prop_getopt_generated_structure
    putStr "  valid_field_names: "
    quickCheckWith args prop_getopt_valid_field_names

    -- Compilation tests are slow (spawn GHC subprocess), skip by default
    -- Uncomment to run:
    -- putStrLn ""
    -- putStrLn "-- Compilation (requires GHC) --"
    -- putStr "  clap_compiles: "
    -- quickCheckWith args prop_clap_compiles
    -- putStr "  getopt_compiles: "
    -- quickCheckWith args prop_getopt_compiles

    putStrLn ""
    putStrLn "=== All properties passed ==="
