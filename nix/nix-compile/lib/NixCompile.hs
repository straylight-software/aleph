{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : NixCompile
-- Description : Top-level API for nix-compile
--
-- Shell scripts as data structures.
--
-- @
-- import NixCompile
--
-- main = do
--   script <- parseScript "PORT=\"\${PORT:-8080}\"\nconfig.server.port=\$PORT"
--   print (scriptSchema script)
-- @
module NixCompile
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
    module NixCompile.Types,
  )
where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import NixCompile.Types
import NixCompile.Bash.Parse (parseBash)
import NixCompile.Bash.Facts (extractFacts)
import NixCompile.Infer.Constraint (factsToConstraints)
import NixCompile.Infer.Unify (solve)
import NixCompile.Schema.Build (buildSchema)

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
