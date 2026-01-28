{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Aleph.Script.Nvidia.Config
Description : Configuration types for NVIDIA SDK extraction

Haskell types for reading Dhall configuration.
-}
module Aleph.Script.Nvidia.Config (
    -- * Types
    NvidiaVersions (..),
    NvidiaSdkConfig (..),

    -- * Loading
    loadConfig,
    defaultConfig,
) where

import Data.Text (Text)
import qualified Data.Text as T
import Dhall (FromDhall, Generic, auto, input)

-- | Expected version numbers
data NvidiaVersions = NvidiaVersions
    { cuda :: Text
    , cudnn :: Text
    , nccl :: Text
    , tensorrt :: Text
    , cutensor :: Text
    }
    deriving (Show, Eq, Generic)

instance FromDhall NvidiaVersions

-- | SDK extraction configuration
data NvidiaSdkConfig = NvidiaSdkConfig
    { imageRef :: Text
    -- ^ Container image to extract
    , outputDir :: Text
    -- ^ Output directory
    , platform :: Text
    -- ^ Target platform
    , expectedVersions :: NvidiaVersions
    -- ^ Expected versions for validation
    , cacheDir :: Text
    -- ^ Container cache directory
    , uploadToR2 :: Bool
    -- ^ Upload to R2
    , r2Bucket :: Maybe Text
    -- ^ R2 bucket
    , r2Endpoint :: Maybe Text
    -- ^ R2 endpoint
    }
    deriving (Show, Eq, Generic)

instance FromDhall NvidiaSdkConfig

-- | Load configuration from a Dhall file
loadConfig :: FilePath -> IO NvidiaSdkConfig
loadConfig path = input auto (T.pack path)

-- | Default configuration for testing
defaultConfig :: NvidiaSdkConfig
defaultConfig =
    NvidiaSdkConfig
        { imageRef = "nvidia/cuda:13.0.1-devel-ubuntu22.04"
        , outputDir = "./nvidia-sdk"
        , platform = "linux/amd64"
        , expectedVersions =
            NvidiaVersions
                { cuda = ""
                , cudnn = ""
                , nccl = ""
                , tensorrt = ""
                , cutensor = ""
                }
        , cacheDir = ""
        , uploadToR2 = False
        , r2Bucket = Nothing
        , r2Endpoint = Nothing
        }
