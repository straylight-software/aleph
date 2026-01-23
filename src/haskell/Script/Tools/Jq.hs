{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Jq
Description : Typed wrapper for jq - JSON processor

This module provides a typed interface to jq. Key features:

* Type-safe options (no typos like @--raw-outut@)
* Proper handling of @--arg@ and @--argjson@ for passing values
* Clean integration with Aleph.Script's Aeson re-exports

@
import Aleph.Script
import qualified Aleph.Script.Tools.Jq as Jq

main = script $ do
  -- Extract a field with raw output
  name <- Jq.jq Jq.defaults { Jq.rawOutput = True } ".name" ["package.json"]

  -- Use --arg to pass a variable safely (shell-safe!)
  result <- Jq.jqWithArgs
    Jq.defaults { Jq.rawOutput = True }
    [Jq.Arg "pattern" searchPattern]
    "select(.name | contains($pattern))"
    ["data.json"]

  -- Parse stdin
  piped <- Jq.jqStdin Jq.defaults ".items[]" jsonText
@
-}
module Aleph.Script.Tools.Jq (
    -- * Options
    Options (..),
    defaults,

    -- * Arguments for --arg/--argjson
    JqArg (..),
    arg,
    argjson,
    slurpfile,
    rawfile,

    -- * Invocation
    jq,
    jq_,
    jqWithArgs,
    jqWithArgs_,
    jqStdin,
    jqStdin_,

    -- * Building args manually
    buildArgs,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options for jq invocation

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { nullInput :: Bool
    -- ^ -n: use null as input instead of reading
    , rawInput :: Bool
    -- ^ -R: read each line as a raw string
    , slurp :: Bool
    -- ^ -s: read all inputs into an array
    , compactOutput :: Bool
    -- ^ -c: compact output (no pretty printing)
    , rawOutput :: Bool
    -- ^ -r: output raw strings (no JSON encoding)
    , rawOutput0 :: Bool
    -- ^ --raw-output0: raw output with NUL separator
    , joinOutput :: Bool
    -- ^ -j: like -r but no newline after each output
    , asciiOutput :: Bool
    -- ^ -a: escape non-ASCII characters
    , sortKeys :: Bool
    -- ^ -S: sort object keys
    , colorOutput :: Bool
    -- ^ -C: colorize output (for terminal)
    , monochromeOutput :: Bool
    -- ^ -M: disable colors
    , tab :: Bool
    -- ^ use tabs for indentation
    , indent :: Maybe Int
    -- ^ number of spaces for indentation (0-7)
    , exitStatus :: Bool
    -- ^ -e: set exit status based on output
    , fromFile :: Maybe FilePath
    -- ^ -f: load filter from file
    }
    deriving (Show, Eq)

-- | Default options (pretty printed JSON output)
defaults :: Options
defaults =
    Options
        { nullInput = False
        , rawInput = False
        , slurp = False
        , compactOutput = False
        , rawOutput = False
        , rawOutput0 = False
        , joinOutput = False
        , asciiOutput = False
        , sortKeys = False
        , colorOutput = False
        , monochromeOutput = False
        , tab = False
        , indent = Nothing
        , exitStatus = False
        , fromFile = Nothing
        }

{- | Arguments passed to jq via --arg, --argjson, etc.

These are the safe way to pass dynamic values into jq filters
without shell escaping issues.
-}
data JqArg
    = -- | --arg NAME VALUE (string)
      Arg Text Text
    | -- | --argjson NAME JSON_VALUE
      ArgJson Text Text
    | -- | --slurpfile NAME FILE (array of JSON from file)
      SlurpFile Text FilePath
    | -- | --rawfile NAME FILE (string contents of file)
      RawFile Text FilePath
    deriving (Show, Eq)

-- | Create a string argument (--arg)
arg :: Text -> Text -> JqArg
arg = Arg

-- | Create a JSON argument (--argjson)
argjson :: Text -> Text -> JqArg
argjson = ArgJson

-- | Create a slurpfile argument (--slurpfile)
slurpfile :: Text -> FilePath -> JqArg
slurpfile = SlurpFile

-- | Create a rawfile argument (--rawfile)
rawfile :: Text -> FilePath -> JqArg
rawfile = RawFile

-- | Build command-line arguments from options
buildArgs :: Options -> [JqArg] -> [Text]
buildArgs Options{..} jqArgs =
    catMaybes
        [ flag nullInput "-n"
        , flag rawInput "-R"
        , flag slurp "-s"
        , flag compactOutput "-c"
        , flag rawOutput "-r"
        , flag rawOutput0 "--raw-output0"
        , flag joinOutput "-j"
        , flag asciiOutput "-a"
        , flag sortKeys "-S"
        , flag colorOutput "-C"
        , flag monochromeOutput "-M"
        , flag tab "--tab"
        , opt indent "--indent"
        , flag exitStatus "-e"
        , optFp fromFile "-f"
        ]
        ++ concatMap argToFlags jqArgs
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just n) f = Just (f <> " " <> pack (show n))
    opt Nothing _ = Nothing
    optFp (Just p) f = Just (f <> " " <> pack p)
    optFp Nothing _ = Nothing

    argToFlags :: JqArg -> [Text]
    argToFlags (Arg name val) = ["--arg", name, val]
    argToFlags (ArgJson name val) = ["--argjson", name, val]
    argToFlags (SlurpFile name fp) = ["--slurpfile", name, pack fp]
    argToFlags (RawFile name fp) = ["--rawfile", name, pack fp]

{- | Run jq with a filter on files, capture output

@
result <- jq defaults { rawOutput = True } ".name" ["package.json"]
@
-}
jq :: Options -> Text -> [FilePath] -> Sh Text
jq opts filterExpr files = run "jq" (buildArgs opts [] ++ [filterExpr] ++ map pack files)

-- | Run jq, ignoring output
jq_ :: Options -> Text -> [FilePath] -> Sh ()
jq_ opts filterExpr files = run_ "jq" (buildArgs opts [] ++ [filterExpr] ++ map pack files)

{- | Run jq with --arg/--argjson arguments

This is the safe way to pass dynamic values into jq:

@
-- SAFE: pattern is properly escaped by jq
jqWithArgs defaults [Arg "pat" userInput] "select(.x | contains($pat))" files

-- UNSAFE: shell injection risk!
bash $ "jq 'select(.x | contains(\"" <> userInput <> "\"))' file"
@
-}
jqWithArgs :: Options -> [JqArg] -> Text -> [FilePath] -> Sh Text
jqWithArgs opts args filterExpr files =
    run "jq" (buildArgs opts args ++ [filterExpr] ++ map pack files)

-- | Run jq with args, ignoring output
jqWithArgs_ :: Options -> [JqArg] -> Text -> [FilePath] -> Sh ()
jqWithArgs_ opts args filterExpr files =
    run_ "jq" (buildArgs opts args ++ [filterExpr] ++ map pack files)

{- | Run jq on text input (piped via stdin simulation)

Note: This writes input to a temp file and reads it, which is safer
than trying to pipe through shell.
-}
jqStdin :: Options -> Text -> Text -> Sh Text
jqStdin opts filterExpr input = withTmpFile $ \tmpFile -> do
    liftIO $ writeFile tmpFile (unpack input)
    jq opts filterExpr [tmpFile]

-- | Run jq on text input, ignoring output
jqStdin_ :: Options -> Text -> Text -> Sh ()
jqStdin_ opts filterExpr input = withTmpFile $ \tmpFile -> do
    liftIO $ writeFile tmpFile (unpack input)
    jq_ opts filterExpr [tmpFile]
