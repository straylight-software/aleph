{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render.Bash.Parse
-- Description : Parse bash scripts using ShellCheck's parser
--
-- Uses ShellCheck as a library to get the bash AST, then walks it
-- to extract facts about variable usage, commands, etc.
module Render.Bash.Parse
  ( parseBash,
    parseBashFile,
    BashAST,
  )
where

import Control.Monad.Identity (Identity, runIdentity)
import qualified Data.Text as T
import Data.Text (Text)
import ShellCheck.AST (Token)
import ShellCheck.Interface
  ( ParseResult (..),
    ParseSpec (..),
    SystemInterface (..),
    newParseSpec,
    newSystemInterface,
  )
import ShellCheck.Parser (parseScript)

-- | The AST from ShellCheck
type BashAST = Token

-- | Parse bash source text
parseBash :: Text -> Either Text BashAST
parseBash src =
  let spec =
        newParseSpec
          { psFilename = "<input>",
            psScript = T.unpack src
          }
      result = runIdentity $ parseScript sysInterface spec
   in case prRoot result of
        Just ast -> Right ast
        Nothing -> Left $ T.pack $ "Parse errors: " ++ show (length (prComments result))
  where
    sysInterface :: SystemInterface Identity
    sysInterface =
      newSystemInterface
        { siReadFile = \_ _ -> return (Left "no file access")
        }

-- | Parse a bash file
parseBashFile :: FilePath -> IO (Either Text BashAST)
parseBashFile path = do
  content <- readFile path
  return $ parseBash (T.pack content)
