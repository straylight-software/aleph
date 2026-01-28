{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Property tests for clap parser using ground truth from Rust fuzzer

Strategy:
  1. Run clap-fuzzer.rs with random seeds to generate help text + ground truth
  2. Parse the help text with our Haskell parser
  3. Compare parsed result against ground truth JSON
-}
module Main where

import Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import Control.Exception (SomeException)
import qualified Control.Exception as E
import Control.Monad (forM, forM_, unless, void, when)
import Data.Aeson (FromJSON (..), eitherDecodeStrict)
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Char8 as BS
import Data.Either (isRight)
import Data.Maybe (catMaybes, isJust, mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Data.Void
import GHC.Generics (Generic)
import System.Exit (ExitCode (..))
import System.Process (readProcess, readProcessWithExitCode)
import Text.Megaparsec
import Text.Megaparsec.Char (char, digitChar, letterChar, string)

-- ============================================================================
-- Ground truth types (from JSON)
-- ============================================================================

data GroundTruth = GroundTruth
    { gtName :: Text
    , gtPositionalArgs :: [PositionalTruth]
    , gtOptions :: [OptionTruth]
    , gtSubcommands :: [SubcommandTruth]
    }
    deriving (Show, Eq, Generic)

data OptionTruth = OptionTruth
    { otShort :: Maybe Char
    , otLong :: Maybe Text
    , otValueName :: Maybe Text
    , otHelp :: Text
    , otRequired :: Bool
    , otTakesValue :: Bool
    , otMultiple :: Bool
    , otEnv :: Maybe Text
    , otDefault :: Maybe Text
    }
    deriving (Show, Eq, Generic)

data SubcommandTruth = SubcommandTruth
    { stName :: Text
    , stAbout :: Text
    }
    deriving (Show, Eq, Generic)

data PositionalTruth = PositionalTruth
    { ptName :: Text
    , ptHelp :: Text
    , ptRequired :: Bool
    , ptMultiple :: Bool
    }
    deriving (Show, Eq, Generic)

instance FromJSON GroundTruth where
    parseJSON = Aeson.withObject "GroundTruth" $ \o ->
        GroundTruth
            <$> o Aeson..: "name"
            <*> o Aeson..: "positional_args"
            <*> o Aeson..: "options"
            <*> o Aeson..: "subcommands"

instance FromJSON PositionalTruth where
    parseJSON = Aeson.withObject "PositionalTruth" $ \o ->
        PositionalTruth
            <$> o Aeson..: "name"
            <*> o Aeson..: "help"
            <*> o Aeson..: "required"
            <*> o Aeson..: "multiple"

instance FromJSON OptionTruth where
    parseJSON = Aeson.withObject "OptionTruth" $ \o ->
        OptionTruth
            <$> o Aeson..: "short"
            <*> o Aeson..: "long"
            <*> o Aeson..: "value_name"
            <*> o Aeson..: "help"
            <*> o Aeson..: "required"
            <*> o Aeson..: "takes_value"
            <*> o Aeson..: "multiple"
            <*> o Aeson..: "env"
            <*> o Aeson..: "default"

instance FromJSON SubcommandTruth where
    parseJSON = Aeson.withObject "SubcommandTruth" $ \o ->
        SubcommandTruth
            <$> o Aeson..: "name"
            <*> o Aeson..: "about"

-- ============================================================================
-- Parsed types (what our parser produces)
-- ============================================================================

data ClapOption = ClapOption
    { optShort :: Maybe Char
    , optLong :: Maybe Text
    , optArg :: Maybe Text
    , optDesc :: Text
    }
    deriving (Show, Eq)

data ClapPositional = ClapPositional
    { posName :: Text
    , posRequired :: Bool -- <NAME> vs [NAME]
    }
    deriving (Show, Eq)

data ClapSection = ClapSection
    { secName :: Text
    , secOptions :: [ClapOption]
    }
    deriving (Show, Eq)

data ClapHelp = ClapHelp
    { helpSections :: [ClapSection]
    , helpPositionals :: [ClapPositional]
    }
    deriving (Show, Eq)

-- ============================================================================
-- Parser (from clap-parser.hs)
-- ============================================================================

type Parser = Parsec Void Text

hspace :: Parser ()
hspace = void $ takeWhileP Nothing (\c -> c == ' ' || c == '\t')

restOfLine :: Parser Text
restOfLine = do
    content <- takeWhileP Nothing (/= '\n')
    void (char '\n') <|> eof
    return content

shortOpt :: Parser Char
shortOpt = char '-' *> (letterChar <|> digitChar <|> char '.')

longOpt :: Parser Text
longOpt = string "--" *> takeWhile1P Nothing isOptChar
  where
    isOptChar c = c == '-' || c == '_' || c `elem` ['a' .. 'z'] || c `elem` ['A' .. 'Z'] || c `elem` ['0' .. '9']

argPlaceholder :: Parser Text
argPlaceholder =
    choice
        [ char '=' *> takeWhile1P Nothing (\c -> c /= ' ' && c /= '\t' && c /= '\n')
        , try $ hspace *> char '<' *> takeWhile1P Nothing (/= '>') <* char '>'
        , char '[' *> (char '=' *> takeWhile1P Nothing (\c -> c /= ']' && c /= '\n')) <* char ']'
        ]

optionLine :: Parser ClapOption
optionLine = do
    hspace
    mShort <- optional $ try (shortOpt <* optional (string ", "))
    mLong <- optional $ try longOpt
    mArg <- optional $ try argPlaceholder
    hspace
    desc <- restOfLine
    case (mShort, mLong) of
        (Nothing, Nothing) -> empty
        _ ->
            return
                ClapOption
                    { optShort = mShort
                    , optLong = mLong
                    , optArg = mArg
                    , optDesc = T.strip desc
                    }

isSectionLine :: Text -> Bool
isSectionLine t =
    let stripped = T.strip t
     in not (T.null stripped)
            && T.last stripped == ':'
            && not (T.isPrefixOf "-" stripped)
            && not (T.isPrefixOf " " t)
            && not (T.isInfixOf "://" stripped)

-- Parse a positional argument line like "  [NAME]" or "  <NAME>"
positionalLine :: Parser ClapPositional
positionalLine = do
    hspace
    (name, req) <-
        choice
            [ do
                -- Required: <NAME>
                _ <- char '<'
                n <- takeWhile1P Nothing (\c -> c /= '>' && c /= '\n')
                _ <- char '>'
                return (n, True)
            , do
                -- Optional: [NAME]
                _ <- char '['
                n <- takeWhile1P Nothing (\c -> c /= ']' && c /= '\n')
                _ <- char ']'
                return (n, False)
            ]
    _ <- restOfLine
    return ClapPositional{posName = name, posRequired = req}

parseHelp :: Text -> ClapHelp
parseHelp input =
    let lns = T.lines input
        (sections, positionals) = go [] [] Nothing lns
     in ClapHelp{helpSections = sections, helpPositionals = positionals}
  where
    go secAcc posAcc _curSec [] =
        case _curSec of
            Just (name, opts) -> (reverse (ClapSection name (reverse opts) : secAcc), reverse posAcc)
            Nothing -> (reverse secAcc, reverse posAcc)
    go secAcc posAcc curSec (line : rest)
        | isSectionLine line =
            let secName = T.dropEnd 1 (T.strip line)
                secAcc' = case curSec of
                    Just (name, opts) -> ClapSection name (reverse opts) : secAcc
                    Nothing -> secAcc
             in go secAcc' posAcc (Just (secName, [])) rest
        | Just (name, opts) <- curSec =
            -- In Arguments section, try to parse positional first
            if name == "Arguments" || name == "POSITIONAL ARGUMENTS"
                then case parse positionalLine "" (line <> "\n") of
                    Right pos -> go secAcc (pos : posAcc) curSec rest
                    Left _ ->
                        if T.null (T.strip line) || T.isPrefixOf "  " line
                            then go secAcc posAcc curSec rest
                            else go (ClapSection name (reverse opts) : secAcc) posAcc Nothing rest
                else case parse optionLine "" (line <> "\n") of
                    Right opt -> go secAcc posAcc (Just (name, opt : opts)) rest
                    Left _ ->
                        -- Skip blank lines and continuation lines (indented)
                        if T.null (T.strip line) || T.isPrefixOf "  " line
                            then go secAcc posAcc curSec rest
                            else go (ClapSection name (reverse opts) : secAcc) posAcc Nothing rest
        | otherwise = go secAcc posAcc curSec rest

-- ============================================================================
-- Comparison logic
-- ============================================================================

-- | Find the Options section from parsed help
getOptionsSection :: ClapHelp -> Maybe ClapSection
getOptionsSection help =
    case filter (\s -> secName s == "Options") (helpSections help) of
        [sec] -> Just sec
        _ -> Nothing

-- | Check if a parsed option matches ground truth option
optionMatches :: ClapOption -> OptionTruth -> Bool
optionMatches parsed truth =
    optShort parsed == otShort truth
        && optLong parsed == otLong truth
        &&
        -- Value name comparison: parsed has it iff truth takes_value
        (isJust (optArg parsed) == otTakesValue truth)
        &&
        -- If both have value names, they should match
        ( case (optArg parsed, otValueName truth) of
            (Just p, Just t) -> p == t
            _ -> True
        )

-- | Find matching ground truth option for a parsed option
findMatch :: ClapOption -> [OptionTruth] -> Maybe OptionTruth
findMatch parsed truths =
    case filter (optionMatches parsed) truths of
        [t] -> Just t
        _ -> Nothing

-- | Comparison result
data CompareResult = CompareResult
    { crOptMatched :: Int
    , crOptMissedFromTruth :: [OptionTruth] -- In truth but not parsed
    , crOptExtraInParsed :: [ClapOption] -- In parsed but not truth (excluding -h, -V)
    , crPosMatched :: Int
    , crPosMissedFromTruth :: [PositionalTruth]
    , crPosExtraInParsed :: [ClapPositional]
    }
    deriving (Show)

-- | Check if positional matches
positionalMatches :: ClapPositional -> PositionalTruth -> Bool
positionalMatches parsed truth =
    posName parsed == ptName truth
        && posRequired parsed == ptRequired truth

-- | Compare parsed options against ground truth
compareOptions :: [ClapOption] -> [OptionTruth] -> ([OptionTruth], [ClapOption], Int)
compareOptions parsed truths =
    let
        -- Filter out help and version from parsed (clap adds these automatically)
        userParsed = filter (\o -> optLong o /= Just "help" && optLong o /= Just "version") parsed

        -- Try to match each parsed option
        matches = [(p, findMatch p truths) | p <- userParsed]
        matched = [(p, t) | (p, Just t) <- matches]
        unmatched = [p | (p, Nothing) <- matches]

        -- Find truth options that weren't matched
        matchedTruths = map snd matched
        missedTruths = filter (`notElem` matchedTruths) truths
     in
        (missedTruths, unmatched, length matched)

-- | Compare parsed positionals against ground truth
comparePositionals :: [ClapPositional] -> [PositionalTruth] -> ([PositionalTruth], [ClapPositional], Int)
comparePositionals parsed truths =
    let matches = [(p, filter (positionalMatches p) truths) | p <- parsed]
        matched = [(p, t) | (p, [t]) <- matches]
        unmatched = [p | (p, []) <- matches]
        matchedTruths = map snd matched
        missedTruths = filter (`notElem` matchedTruths) truths
     in (missedTruths, unmatched, length matched)

-- | Full comparison
compareAll :: ClapHelp -> GroundTruth -> CompareResult
compareAll parsed truth =
    let optSection = filter (\s -> secName s == "Options") (helpSections parsed)
        parsedOpts = concatMap secOptions optSection
        parsedPos = helpPositionals parsed

        (optMissed, optExtra, optMatched) = compareOptions parsedOpts (gtOptions truth)
        (posMissed, posExtra, posMatched) = comparePositionals parsedPos (gtPositionalArgs truth)
     in CompareResult
            { crOptMatched = optMatched
            , crOptMissedFromTruth = optMissed
            , crOptExtraInParsed = optExtra
            , crPosMatched = posMatched
            , crPosMissedFromTruth = posMissed
            , crPosExtraInParsed = posExtra
            }

isGoodResult :: CompareResult -> Bool
isGoodResult cr =
    null (crOptMissedFromTruth cr)
        && null (crOptExtraInParsed cr)
        && null (crPosMissedFromTruth cr)
        && null (crPosExtraInParsed cr)

-- ============================================================================
-- Fuzzer integration
-- ============================================================================

-- | Run the Rust fuzzer and get output
runFuzzer :: Int -> Bool -> IO (Either String (GroundTruth, Text))
runFuzzer seed useShort = do
    let args = ["clap-fuzzer.rs", "--seed", show seed] ++ ["--short" | useShort]
    result <- E.try $ readProcessWithExitCode "rust-script" args ""
    case result of
        Left (e :: SomeException) -> return $ Left $ "Failed to run fuzzer: " ++ show e
        Right (ExitSuccess, stdout, _stderr) -> do
            let lns = lines stdout
            case lns of
                (jsonLine : "---" : helpLines) ->
                    case eitherDecodeStrict (BS.pack jsonLine) of
                        Left err -> return $ Left $ "JSON parse error: " ++ err
                        Right truth -> return $ Right (truth, T.pack $ unlines helpLines)
                _ -> return $ Left $ "Unexpected fuzzer output format"
        Right (ExitFailure code, _, stderr) ->
            return $ Left $ "Fuzzer failed with code " ++ show code ++ ": " ++ stderr

-- | Test a single seed
testSeed :: Int -> Bool -> IO (Either String CompareResult)
testSeed seed useShort = do
    fuzzerResult <- runFuzzer seed useShort
    case fuzzerResult of
        Left err -> return $ Left err
        Right (truth, helpText) -> do
            let parsed = parseHelp helpText
            return $ Right $ compareAll parsed truth

-- ============================================================================
-- Generators for pure Haskell testing (fallback)
-- ============================================================================

genShortChar :: Gen Char
genShortChar = Gen.element $ ['a' .. 'z'] ++ ['A' .. 'Z'] ++ ['0' .. '9']

genLongName :: Gen Text
genLongName = do
    first <- Gen.element ['a' .. 'z']
    rest <- Gen.text (Range.linear 1 15) (Gen.element $ ['a' .. 'z'] ++ ['-'])
    let name = T.cons first rest
        cleaned = T.replace "--" "-" name
    return $ T.dropWhileEnd (== '-') cleaned

genArgName :: Gen Text
genArgName = do
    first <- Gen.element ['A' .. 'Z']
    rest <- Gen.text (Range.linear 0 8) (Gen.element $ ['A' .. 'Z'] ++ ['_'])
    return $ T.cons first rest

genDesc :: Gen Text
genDesc = do
    words' <- Gen.list (Range.linear 1 10) genWord
    return $ T.intercalate " " words'
  where
    genWord = Gen.text (Range.linear 1 10) (Gen.element $ ['a' .. 'z'] ++ ['A' .. 'Z'])

genOption :: Gen ClapOption
genOption = do
    hasShort <- Gen.bool
    hasLong <- Gen.bool
    let hasShort' = hasShort || not hasLong

    mShort <- if hasShort' then Just <$> genShortChar else pure Nothing
    mLong <- if hasLong || not hasShort' then Just <$> genLongName else pure Nothing
    mArg <- Gen.maybe genArgName
    desc <- genDesc

    return
        ClapOption
            { optShort = mShort
            , optLong = mLong
            , optArg = mArg
            , optDesc = desc
            }

genSectionName :: Gen Text
genSectionName = do
    words' <- Gen.list (Range.linear 1 3) genUpperWord
    return $ T.intercalate " " words'
  where
    genUpperWord = Gen.text (Range.linear 2 10) (Gen.element ['A' .. 'Z'])

genSection :: Gen ClapSection
genSection = do
    name <- genSectionName
    opts <- Gen.list (Range.linear 1 10) genOption
    return ClapSection{secName = name, secOptions = opts}

genHelp :: Gen ClapHelp
genHelp = do
    sections <- Gen.list (Range.linear 1 5) genSection
    return ClapHelp{helpSections = sections, helpPositionals = []}

-- ============================================================================
-- Renderer (for roundtrip testing)
-- ============================================================================

renderHelp :: ClapHelp -> Text
renderHelp ClapHelp{..} = T.unlines $ concatMap renderSection helpSections

-- Note: We don't render positionals in roundtrip tests since they use the Options format

renderSection :: ClapSection -> [Text]
renderSection ClapSection{..} =
    [secName <> ":"] ++ map renderOption secOptions

renderOption :: ClapOption -> Text
renderOption ClapOption{..} =
    let short = maybe "" (\c -> "-" <> T.singleton c) optShort
        long = maybe "" ("--" <>) optLong
        arg = maybe "" ("=" <>) optArg
        flags = case (optShort, optLong) of
            (Just _, Just l) -> short <> ", --" <> l <> arg
            (Just _, Nothing) -> short <> arg
            (Nothing, Just l) -> "--" <> l <> arg
            (Nothing, Nothing) -> ""
        padding = T.replicate (max 1 (30 - T.length flags)) " "
     in "  " <> flags <> padding <> optDesc

-- ============================================================================
-- Properties
-- ============================================================================

normalize :: ClapHelp -> ClapHelp
normalize h = h{helpSections = map normSec (helpSections h)}
  where
    normSec s = s{secOptions = map normOpt (secOptions s)}
    normOpt o = o{optDesc = T.strip (optDesc o)}

-- | Roundtrip: generate -> render -> parse -> compare
prop_roundtrip :: Property
prop_roundtrip = withTests 500 $ property $ do
    help <- forAll genHelp
    let rendered = renderHelp help
        parsed = parseHelp rendered
    normalize help === normalize parsed

-- | Parse random unicode without crashing
prop_parse_nocrash :: Property
prop_parse_nocrash = withTests 500 $ property $ do
    text <- forAll $ Gen.text (Range.linear 0 2000) Gen.unicode
    let result = parseHelp text
    assert $ length (helpSections result) >= 0

-- | All parsed options have at least short or long
prop_options_valid :: Property
prop_options_valid = withTests 500 $ property $ do
    help <- forAll genHelp
    let rendered = renderHelp help
        parsed = parseHelp rendered
    forM_ (helpSections parsed) $ \sec ->
        forM_ (secOptions sec) $ \opt ->
            assert $ isJust (optShort opt) || isJust (optLong opt)

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = do
    putStrLn "=== Clap Parser Tests ==="
    putStrLn ""

    -- Run Hedgehog roundtrip tests first
    putStrLn "--- Hedgehog Roundtrip Tests ---"
    roundtripOk <-
        checkParallel $
            Group
                "Roundtrip"
                [ ("prop_roundtrip", prop_roundtrip)
                , ("prop_parse_nocrash", prop_parse_nocrash)
                , ("prop_options_valid", prop_options_valid)
                ]
    putStrLn ""

    -- Now test against real clap output
    putStrLn "--- Fuzzer Integration Tests ---"
    putStrLn "Testing parser against real clap output..."
    putStrLn ""

    -- Test a range of seeds
    let seeds = [1 .. 500]
    results <- forM seeds $ \seed -> do
        shortResult <- testSeed seed True
        longResult <- testSeed seed False
        return (seed, shortResult, longResult)

    -- Summarize results
    let failures =
            [ (s, sr, lr)
            | (s, sr, lr) <- results
            , not (either (const True) isGoodResult sr)
                || not (either (const True) isGoodResult lr)
            ]

        errors =
            [(s, e) | (s, Left e, _) <- results]
                ++ [(s, e) | (s, _, Left e) <- results]

    putStrLn $ "Tested " ++ show (length seeds) ++ " seeds (short + long help)"
    putStrLn $ "Errors (fuzzer/setup): " ++ show (length errors)
    putStrLn $ "Parse mismatches: " ++ show (length failures - length errors)
    putStrLn ""

    -- Show some failures
    when (not (null failures)) $ do
        putStrLn "=== Sample Failures ==="
        forM_ (take 3 failures) $ \(seed, shortRes, longRes) -> do
            putStrLn $ "Seed " ++ show seed ++ ":"
            let showResult label cr = do
                    putStrLn $
                        "  "
                            ++ label
                            ++ ": opts="
                            ++ show (crOptMatched cr)
                            ++ " pos="
                            ++ show (crPosMatched cr)
                    when (not (null (crOptMissedFromTruth cr))) $ do
                        putStrLn $ "    Opts missed from truth:"
                        forM_ (crOptMissedFromTruth cr) $ \t ->
                            putStrLn $
                                "      "
                                    ++ show (otShort t)
                                    ++ " / "
                                    ++ show (otLong t)
                                    ++ " takes_value="
                                    ++ show (otTakesValue t)
                    when (not (null (crOptExtraInParsed cr))) $ do
                        putStrLn $ "    Opts extra in parsed:"
                        forM_ (crOptExtraInParsed cr) $ \p ->
                            putStrLn $ "      " ++ show (optShort p) ++ " / " ++ show (optLong p)
                    when (not (null (crPosMissedFromTruth cr))) $ do
                        putStrLn $ "    Positionals missed from truth:"
                        forM_ (crPosMissedFromTruth cr) $ \t ->
                            putStrLn $ "      " ++ T.unpack (ptName t) ++ " required=" ++ show (ptRequired t)
                    when (not (null (crPosExtraInParsed cr))) $ do
                        putStrLn $ "    Positionals extra in parsed:"
                        forM_ (crPosExtraInParsed cr) $ \p ->
                            putStrLn $ "      " ++ T.unpack (posName p) ++ " required=" ++ show (posRequired p)
            case shortRes of
                Left err -> putStrLn $ "  Short: ERROR - " ++ err
                Right cr -> unless (isGoodResult cr) $ showResult "Short" cr
            case longRes of
                Left err -> putStrLn $ "  Long: ERROR - " ++ err
                Right cr -> unless (isGoodResult cr) $ showResult "Long" cr
            putStrLn ""

    -- Final verdict
    putStrLn "=== Summary ==="
    if roundtripOk && null failures
        then putStrLn "All tests passed!"
        else do
            unless roundtripOk $ putStrLn "Hedgehog tests FAILED"
            unless (null failures) $ putStrLn $ "Fuzzer tests: " ++ show (length failures) ++ " failures"
            putStrLn "SOME TESTS FAILED"
