{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- | Aleph.Build.Context
-- BuildContext read from Dhall. No env vars.
module Aleph.Build.Context
  ( BuildContext (..)
  , Ctx (..)
  , readContext
  , toCtx
  ) where

import Aleph.Build.Triple (Triple)

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Dhall (FromDhall, auto, input)
import GHC.Generics (Generic)
import GHC.Natural (Natural)

-- | The build context. Only input to builders.
-- Matches Drv/BuildContext.dhall exactly.
data BuildContext = BuildContext
  { out :: Text
  , src :: Text
  , host :: Triple
  , target :: Maybe Triple
  , cores :: Natural
  , deps :: Map Text Text
  , specName :: Text
  , specVersion :: Text
  }
  deriving (Show, Generic, FromDhall)

-- | Read BuildContext from a Dhall file
readContext :: FilePath -> IO BuildContext
readContext path = do
  txt <- TIO.readFile path
  input auto txt

-- | Convert BuildContext to the legacy Ctx type (for compatibility)
-- Eventually Ctx goes away and we use BuildContext directly
toCtx :: BuildContext -> Ctx
toCtx BuildContext {..} =
  Ctx
    { ctxOut = T.unpack out
    , ctxSrc = T.unpack src
    , ctxHost = host
    , ctxTarget = target
    , ctxCores = fromIntegral cores
    , ctxDeps = Map.mapKeys T.unpack $ Map.map T.unpack deps
    }

-- | Legacy Ctx type (to be removed)
data Ctx = Ctx
  { ctxOut :: FilePath
  , ctxSrc :: FilePath
  , ctxDeps :: Map String FilePath
  , ctxHost :: Triple
  , ctxTarget :: Maybe Triple
  , ctxCores :: Int
  }
  deriving (Show)
