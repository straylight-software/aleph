{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Crane
Description : Typed wrapper for crane - OCI image tool

Crane is the go-to tool for OCI image operations. This module provides
typed interfaces for common operations:

* Pulling/pushing images
* Exporting image filesystems
* Inspecting image configuration
* Listing tags

@
import Aleph.Script
import qualified Aleph.Script.Tools.Crane as Crane

main = script $ do
  -- Export image to directory
  Crane.export_ Crane.defaults "docker.io/library/alpine:latest" "/tmp/rootfs"

  -- Get image config as JSON
  configJson <- Crane.config "docker.io/library/alpine:latest"
  case eitherDecode (encodeUtf8 configJson) of
    Right cfg -> -- use typed config
    Left err  -> die $ "Failed to parse config: " <> pack err
@
-}
module Aleph.Script.Tools.Crane (
    -- * Options
    Options (..),
    Platform (..),
    defaults,
    defaultPlatform,
    platform,

    -- * Image operations
    export,
    export_,
    exportToDir,
    config,
    manifest,
    digest,
    ls,
    copy,
    copy_,

    -- * Low-level
    crane,
    crane_,
    buildArgs,
) where

import Aleph.Script hiding (FilePath, ls)
import qualified Aleph.Script as WS
import Data.Maybe (catMaybes)

-- | Platform specification for multi-arch images
data Platform = Platform
    { os :: Text
    -- ^ Operating system (e.g., "linux")
    , arch :: Text
    -- ^ Architecture (e.g., "amd64", "arm64")
    }
    deriving (Show, Eq)

-- | Default platform (linux/amd64)
defaultPlatform :: Platform
defaultPlatform = Platform "linux" "amd64"

-- | Create a platform string for crane
platform :: Platform -> Text
platform Platform{..} = os <> "/" <> arch

-- | Common options for crane commands
data Options = Options
    { insecure :: Bool
    -- ^ Allow insecure (HTTP) registries
    , verbose :: Bool
    -- ^ Verbose output
    , platform_ :: Maybe Platform
    -- ^ Target platform for multi-arch images
    }
    deriving (Show, Eq)

-- | Default options (secure, quiet, auto platform)
defaults :: Options
defaults =
    Options
        { insecure = False
        , verbose = False
        , platform_ = Just defaultPlatform
        }

-- | Build common arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag insecure "--insecure"
        , flag verbose "-v"
        , opt (platform <$> platform_) "--platform"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing

-- | Run arbitrary crane command
crane :: Options -> Text -> [Text] -> Sh Text
crane opts cmd args = run "crane" ([cmd] ++ buildArgs opts ++ args)

-- | Run crane command, ignoring output
crane_ :: Options -> Text -> [Text] -> Sh ()
crane_ opts cmd args = run_ "crane" ([cmd] ++ buildArgs opts ++ args)

{- | Export an image filesystem to stdout (as tar stream)

Returns the tar data as Text (you probably want 'exportToDir' instead)
-}
export :: Options -> Text -> Sh Text
export opts image = crane opts "export" [image, "-"]

{- | Export an image filesystem to stdout, pipe to tar for extraction

This is the common pattern: pull image and extract to a directory.

@
exportToDir defaults "alpine:latest" "/tmp/rootfs"
@
-}
exportToDir :: Options -> Text -> FilePath -> Sh ()
exportToDir opts image destDir = do
    mkdirP destDir
    -- Use shell pipeline for efficiency (don't buffer whole image in memory)
    let platformArg = case platform_ opts of
            Just p -> " --platform " <> platform p
            Nothing -> ""
    bash_ $ "crane export" <> platformArg <> " '" <> image <> "' - | tar -xf - -C " <> pack destDir

-- | Export image, ignoring output (use 'exportToDir' for real work)
export_ :: Options -> Text -> Sh ()
export_ opts image = crane_ opts "export" [image, "-"]

{- | Get image configuration JSON

The returned JSON contains:
* @.config.Env@ - environment variables
* @.config.Cmd@ - default command
* @.config.Entrypoint@ - entrypoint
* @.config.WorkingDir@ - working directory
* @.config.Labels@ - image labels

@
configJson <- config "alpine:latest"
-- Parse with aeson to extract fields
@
-}
config :: Text -> Sh Text
config image = run "crane" ["config", image]

-- | Get image manifest JSON
manifest :: Text -> Sh Text
manifest image = run "crane" ["manifest", image]

-- | Get image digest (sha256)
digest :: Text -> Sh Text
digest image = strip <$> run "crane" ["digest", image]

{- | List tags for a repository

@
tags <- ls "docker.io/library/alpine"
-- Returns: ["3.18", "3.19", "latest", ...]
@
-}
ls :: Text -> Sh [Text]
ls repo = WS.lines . strip <$> run "crane" ["ls", repo]

{- | Copy image between registries

@
copy defaults "docker.io/library/alpine:latest" "ghcr.io/myorg/alpine:latest"
@
-}
copy :: Options -> Text -> Text -> Sh Text
copy opts src dst = crane opts "copy" [src, dst]

-- | Copy image, ignoring output
copy_ :: Options -> Text -> Text -> Sh ()
copy_ opts src dst = crane_ opts "copy" [src, dst]
