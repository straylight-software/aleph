{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render
-- Description : Top-level API for render.nix
--
-- Shell scripts as data structures.
--
-- @
-- import Render
--
-- main = do
--   script <- parseScript "PORT=\"\${PORT:-8080}\"\nconfig.server.port=\$PORT"
--   print (scriptSchema script)
-- @
module Render
  ( -- * Parsing
    parseScript,
    parseScriptFile,

    -- * Schema
    Schema (..),
    EnvSpec (..),
    ConfigSpec (..),
    CommandSpec (..),

    -- * Types
    Type (..),
    Literal (..),
    StorePath (..),

    -- * Errors
    TypeError (..),
    LintError (..),
    Severity (..),

    -- * Re-exports
    module Render.Types,
  )
where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Render.Types
import Render.Bash.Parse (parseBash)
import Render.Bash.Facts (extractFacts)
import Render.Infer.Constraint (factsToConstraints)
import Render.Infer.Unify (solve)
import Render.Schema.Build (buildSchema)

-- | Parse a bash script and extract its schema
parseScript :: Text -> Either Text Script
parseScript src = do
  ast <- parseBash src
  let facts = extractFacts ast
  let constraints = factsToConstraints facts
  subst <- case solve constraints of
    Left err -> Left (T.pack (show err))
    Right s -> Right s
  let schema = buildSchema facts subst
  Right Script
    { scriptSource = src
    , scriptFacts = facts
    , scriptSchema = schema
    }

-- | Parse a bash script file
parseScriptFile :: FilePath -> IO (Either Text Script)
parseScriptFile path = do
  src <- TIO.readFile path
  return (parseScript src)
