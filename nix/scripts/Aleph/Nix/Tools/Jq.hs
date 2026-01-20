{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Typed jq actions for build phases.

Pure data structures describing jq invocations. No runtime dependencies.

@
import qualified Aleph.Nix.Tools.Jq as Jq

postInstall
    [ Jq.query Jq.defaults { Jq.rawOutput = True } ".name" "package.json"
    ]
@

The jq package is automatically added to nativeBuildInputs.
-}
module Aleph.Nix.Tools.Jq (
    -- * Options
    Options (..),
    defaults,

    -- * Build phase actions
    query,
    queryFiles,
    transform,
) where

import Aleph.Nix.Derivation (Action (..))
import Aleph.Nix.Types (PkgRef (..))
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T

-- | Options for jq invocation
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
    , sortKeys :: Bool
    -- ^ -S: sort object keys
    , exitStatus :: Bool
    -- ^ -e: set exit status based on output
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
        , sortKeys = False
        , exitStatus = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag nullInput "-n"
        , flag rawInput "-R"
        , flag slurp "-s"
        , flag compactOutput "-c"
        , flag rawOutput "-r"
        , flag sortKeys "-S"
        , flag exitStatus "-e"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing

-- | The jq package reference
jqPkg :: PkgRef
jqPkg = PkgRef "jq"

{- | Query a single file with jq.

@
Jq.query Jq.defaults { Jq.rawOutput = True } ".name" "package.json"
@
-}
query :: Options -> Text -> Text -> Action
query opts filterExpr file =
    ToolRun jqPkg (buildArgs opts ++ [filterExpr, file])

{- | Query multiple files with jq.

@
Jq.queryFiles Jq.defaults ".version" ["pkg1.json", "pkg2.json"]
@
-}
queryFiles :: Options -> Text -> [Text] -> Action
queryFiles opts filterExpr files =
    ToolRun jqPkg (buildArgs opts ++ [filterExpr] ++ files)

{- | Transform a file in-place.

Uses a temp file to avoid truncation issues.

@
Jq.transform Jq.defaults ".version = \"2.0\"" "package.json"
@
-}
transform :: Options -> Text -> Text -> Action
transform opts filterExpr file =
    Run "sh" ["-c", cmd]
  where
    argsStr = T.unwords (buildArgs opts)
    cmd =
        "jq "
            <> argsStr
            <> " "
            <> filterExpr
            <> " "
            <> file
            <> " > "
            <> file
            <> ".tmp && mv "
            <> file
            <> ".tmp "
            <> file
