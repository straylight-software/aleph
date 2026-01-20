{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Parser for clap (Rust) -h/--help output

Clap is the most common CLI library in the Rust ecosystem, used by:
  ripgrep, fd, bat, hyperfine, tokei, dust, exa
  cloud-hypervisor, mdbook, nixdoc, ndg, deadnix, statix
  ruff, biome, stylua, taplo

Works best with short help (-h), handles:
  -s, --long=VALUE    Description
  --long <value>      Description
  -x                  Short only
-}
module Main where

import Control.Monad (void)
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Data.Void
import System.Environment (getArgs)
import Text.Megaparsec
import Text.Megaparsec.Char hiding (hspace, hspace1)

type Parser = Parsec Void Text

-- | A CLI option/flag
data ClapOption = ClapOption
    { optShort :: Maybe Char
    , optLong :: Maybe Text
    , optArg :: Maybe Text
    , optDesc :: Text
    }
    deriving (Show, Eq)

-- | A section of options
data ClapSection = ClapSection
    { secName :: Text
    , secOptions :: [ClapOption]
    }
    deriving (Show, Eq)

-- | Full parsed help
data ClapHelp = ClapHelp
    { helpSections :: [ClapSection]
    }
    deriving (Show, Eq)

-- Horizontal whitespace
hspace :: Parser ()
hspace = void $ takeWhileP Nothing (\c -> c == ' ' || c == '\t')

-- Rest of line
restOfLine :: Parser Text
restOfLine = do
    content <- takeWhileP Nothing (/= '\n')
    void (char '\n') <|> eof
    return content

-- Parse short option: -x or -.
shortOpt :: Parser Char
shortOpt = char '-' *> (letterChar <|> digitChar <|> char '.')

-- Parse long option: --name
longOpt :: Parser Text
longOpt = string "--" *> takeWhile1P Nothing isOptChar
  where
    isOptChar c = c == '-' || c == '_' || c `elem` ['a' .. 'z'] || c `elem` ['A' .. 'Z'] || c `elem` ['0' .. '9']

-- Parse argument: =VAL or <val> or space <val>
argPlaceholder :: Parser Text
argPlaceholder =
    choice
        [ char '=' *> takeWhile1P Nothing (\c -> c /= ' ' && c /= '\t' && c /= '\n')
        , try $ hspace *> char '<' *> takeWhile1P Nothing (/= '>') <* char '>'
        , char '[' *> (char '=' *> takeWhile1P Nothing (\c -> c /= ']' && c /= '\n')) <* char ']' -- [=val]
        ]

-- Parse one option line
optionLine :: Parser ClapOption
optionLine = do
    hspace
    -- Short option
    mShort <- optional $ try (shortOpt <* optional (string ", "))
    -- Long option
    mLong <- optional $ try longOpt
    -- Argument
    mArg <- optional $ try argPlaceholder
    -- Description (rest of line)
    hspace
    desc <- restOfLine
    -- Must have at least short or long
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

-- Check if line looks like section header (ends with :, not indented, not an option)
isSectionLine :: Text -> Bool
isSectionLine t =
    let stripped = T.strip t
     in not (T.null stripped)
            && T.last stripped == ':'
            && not (T.isPrefixOf "-" stripped)
            && not (T.isPrefixOf " " t) -- not indented
            && not (T.isInfixOf "://" stripped) -- not a URL

-- Parse all content, extracting sections
parseHelp :: Text -> ClapHelp
parseHelp input = ClapHelp{helpSections = go [] Nothing (T.lines input)}
  where
    go acc _curSec [] =
        case _curSec of
            Just (name, opts) -> reverse (ClapSection name (reverse opts) : acc)
            Nothing -> reverse acc
    go acc curSec (line : rest)
        | isSectionLine line =
            let secName = T.dropEnd 1 (T.strip line)
                acc' = case curSec of
                    Just (name, opts) -> ClapSection name (reverse opts) : acc
                    Nothing -> acc
             in go acc' (Just (secName, [])) rest
        | Just (name, opts) <- curSec =
            case parse optionLine "" (line <> "\n") of
                Right opt -> go acc (Just (name, opt : opts)) rest
                Left _ ->
                    -- Skip blank lines and continuation lines (indented)
                    if T.null (T.strip line) || T.isPrefixOf "  " line
                        then go acc curSec rest
                        else go (ClapSection name (reverse opts) : acc) Nothing rest
        | otherwise = go acc curSec rest

-- Pretty print
ppClapHelp :: ClapHelp -> Text
ppClapHelp ClapHelp{..} = T.unlines $ concatMap ppSection helpSections
  where
    ppSection ClapSection{..} =
        [secName <> ":"] ++ map ppOpt secOptions ++ [""]
    ppOpt ClapOption{..} =
        let short = maybe "" (\c -> "-" <> T.singleton c) optShort
            long = maybe "" ("--" <>) optLong
            arg = maybe "" (\a -> "=" <> a) optArg
            flags = T.intercalate ", " $ filter (not . T.null) [short, long <> arg]
         in "  " <> flags <> ": " <> optDesc

main :: IO ()
main = do
    args <- getArgs
    input <- case args of
        [] -> TIO.getContents
        [f] -> TIO.readFile f
        _ -> error "Usage: clap-parser [FILE]"
    let help = parseHelp input
        total = sum $ map (length . secOptions) (helpSections help)
    TIO.putStr $ ppClapHelp help
    putStrLn $
        "Parsed "
            ++ show total
            ++ " options in "
            ++ show (length (helpSections help))
            ++ " sections"
