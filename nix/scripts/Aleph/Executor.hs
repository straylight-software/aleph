{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Executor Abstraction

An Executor provides the runtime capabilities for building:
- Fetching sources
- Resolving dependencies  
- Store operations

This abstraction allows the same builder code to run on:
- Nix (via WASI FFI to straylight-nix)
- Buck2 (via WASI FFI to buck2 host)
- Standalone (via local implementations)

= Design

The builder is a pure function: Spec -> [Action]
The executor interprets those actions.

But for Aleph-1, we go further: the builder CALLS the executor
to fetch and resolve, then runs the build directly. This gives
Haskell control of the orchestration.

@
                    ┌─────────────────┐
                    │   Dhall Spec    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Haskell Builder │
                    │                 │
                    │  1. Parse spec  │
                    │  2. Call exec   │◄───┐
                    │  3. Build       │    │
                    └────────┬────────┘    │
                             │             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   Nix    │   │  Buck2   │   │Standalone│
       │ Executor │   │ Executor │   │ Executor │
       └──────────┘   └──────────┘   └──────────┘
@

= Executor Interface

Every executor must provide:

1. **Fetch**: Get sources into a content-addressed store
2. **Resolve**: Map dependency names to store paths
3. **Context**: Provide build context (output path, cores, system)

The interface is defined by WASI imports, but we also provide
a Haskell typeclass for testing and native execution.
-}
module Aleph.Executor (
    -- * The Executor typeclass
    Executor (..),
    
    -- * Source types
    Source (..),
    
    -- * Build context
    BuildEnv (..),
    
    -- * Default executor (WASI FFI)
    wasiExecutor,
) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import qualified Aleph.Nix.Fetch as FFI
import qualified Aleph.Nix.Store as FFI

-- | Source specification (matches Dhall Src type)
data Source
    = GitHub { srcOwner :: Text, srcRepo :: Text, srcRev :: Text, srcHash :: Text }
    | Url { srcUrl :: Text, srcHash :: Text }
    | Git { srcGitUrl :: Text, srcGitRev :: Text, srcGitHash :: Text }
    | Local { srcPath :: Text }
    | NoSource
    deriving (Show, Eq)

-- | Build environment provided by the executor
data BuildEnv = BuildEnv
    { envOut :: FilePath       -- ^ Output path ($out)
    , envSrc :: FilePath       -- ^ Source path
    , envDeps :: Map Text FilePath  -- ^ Resolved dependencies
    , envSystem :: Text        -- ^ System string
    , envCores :: Int          -- ^ Available cores
    }
    deriving (Show)

-- | The Executor typeclass - what any executor must provide
class Monad m => Executor m where
    -- | Fetch a source, return store path
    fetch :: Source -> m FilePath
    
    -- | Resolve a dependency name to store path
    resolve :: Text -> m FilePath
    
    -- | Get the output path for a named output
    getOutput :: Text -> m FilePath
    
    -- | Get system string
    getSystemString :: m Text
    
    -- | Get number of cores
    getCoreCount :: m Int
    
    -- | Build the full environment
    buildEnv :: Source -> [Text] -> m BuildEnv
    buildEnv src depNames = do
        srcPath <- fetch src
        deps <- Map.fromList <$> mapM (\n -> (n,) <$> resolve n) depNames
        outPath <- getOutput "out"
        system <- getSystemString
        cores <- getCoreCount
        pure BuildEnv
            { envOut = T.unpack outPath
            , envSrc = srcPath
            , envDeps = Map.map T.unpack deps
            , envSystem = system
            , envCores = cores
            }

-- | The WASI executor - uses FFI to call the host
instance Executor IO where
    fetch (GitHub owner repo rev hash) = T.unpack <$> FFI.fetchGitHub owner repo rev hash
    fetch (Url url hash) = T.unpack <$> FFI.fetchUrl url hash
    fetch (Git url rev hash) = T.unpack <$> FFI.fetchGit url rev hash
    fetch (Local path) = pure (T.unpack path)
    fetch NoSource = pure ""
    
    resolve name = T.unpack <$> FFI.resolveDep name
    
    getOutput name = FFI.getOutPath name
    
    getSystemString = FFI.getSystem
    
    getCoreCount = FFI.getCores

-- | Convenience: the default WASI executor
wasiExecutor :: IO BuildEnv -> IO BuildEnv
wasiExecutor = id  -- IO is already the WASI executor
