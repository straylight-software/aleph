{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Run NativeLink containers locally for development/testing.

Usage: nativelink-local [OPTIONS] [COMPONENT]

Components:
  scheduler   Run scheduler only
  cas         Run CAS only  
  worker      Run worker only
  all         Run all components (default)
  shell       Shell into a running container

Options:
  --fc            Use Firecracker microVM (default)
  --oci           Use OCI namespace (bubblewrap)
  --image NAME    Override container image
  --cpus N        vCPU count (default: 2)
  --mem N         Memory in MiB (default: 1024)

Example:
  nativelink-local                    # Run all in Firecracker
  nativelink-local --oci scheduler    # Run scheduler in namespace
  nativelink-local shell worker       # Shell into worker
-}
module Main where

import Aleph.Script
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Data.Text as T
import Data.Function ((&))
import System.Environment (getArgs)

data Runtime = Firecracker | OciNamespace
    deriving (Eq, Show)

data Config = Config
    { cfgRuntime :: Runtime
    , cfgCpus :: Int
    , cfgMemMib :: Int
    , cfgComponent :: Component
    , cfgImageOverride :: Maybe Text
    }

data Component = All | Scheduler | Cas | Worker | Shell Text
    deriving (Eq, Show)

defaultConfig :: Config
defaultConfig = Config
    { cfgRuntime = Firecracker
    , cfgCpus = 2
    , cfgMemMib = 1024
    , cfgComponent = All
    , cfgImageOverride = Nothing
    }

parseArgs :: [String] -> Config
parseArgs = go defaultConfig
  where
    go cfg [] = cfg
    go cfg ("--fc" : rest) = go cfg { cfgRuntime = Firecracker } rest
    go cfg ("--oci" : rest) = go cfg { cfgRuntime = OciNamespace } rest
    go cfg ("--cpus" : n : rest) = go cfg { cfgCpus = read n } rest
    go cfg ("--mem" : n : rest) = go cfg { cfgMemMib = read n } rest
    go cfg ("--image" : i : rest) = go cfg { cfgImageOverride = Just (pack i) } rest
    go cfg ("scheduler" : rest) = go cfg { cfgComponent = Scheduler } rest
    go cfg ("cas" : rest) = go cfg { cfgComponent = Cas } rest
    go cfg ("worker" : rest) = go cfg { cfgComponent = Worker } rest
    go cfg ("all" : rest) = go cfg { cfgComponent = All } rest
    go cfg ("shell" : name : rest) = go cfg { cfgComponent = Shell (pack name) } rest
    go cfg (_ : rest) = go cfg rest

-- | Get the nix package output path for a container
getContainerImage :: Text -> Sh Text
getContainerImage name = do
    -- Build the container and get its store path
    result <- run "nix" ["build", ".#" <> name, "--print-out-paths", "--no-link"]
    pure $ strip result

-- | Run container with Firecracker
runFirecracker :: Config -> Text -> [Text] -> Sh ()
runFirecracker cfg name cmd = do
    echo $ ":: Starting " <> name <> " in Firecracker microVM"
    imagePath <- getContainerImage ("nativelink-" <> name)
    
    -- isospin-run expects OCI image reference, but we have nix2container output
    -- For local dev, use unshare-run instead or convert
    run_ "isospin-run" $ 
        [ "--cpus", pack (show $ cfgCpus cfg)
        , "--mem", pack (show $ cfgMemMib cfg)
        ] ++ [imagePath] ++ cmd

-- | Run container with OCI namespace (bubblewrap)
runOci :: Config -> Text -> [Text] -> Sh ()
runOci cfg name cmd = do
    echo $ ":: Starting " <> name <> " in OCI namespace"
    
    -- Get the container image path (unused for now, using image name)
    _ <- getContainerImage ("nativelink-" <> name)
    
    -- nix2container outputs are JSON manifests, extract rootfs
    let image = fromMaybe ("nativelink-" <> name) (cfgImageOverride cfg)
    
    -- Pull or use cached
    rootfs <- Oci.pullOrCache Oci.defaultConfig image
    
    -- Build sandbox with extra mounts for nativelink
    let sandbox = Oci.baseSandbox rootfs
            & Bwrap.bind "/tmp/nativelink" "/tmp/nativelink"
            & Bwrap.bind "/data" "/data"
    
    Bwrap.exec sandbox (if Prelude.null cmd then ["/bin/bash"] else cmd)

-- | Run a component
runComponent :: Config -> Text -> Sh ()
runComponent cfg name = do
    let runner = case cfgRuntime cfg of
            Firecracker -> runFirecracker
            OciNamespace -> runOci
    runner cfg name []

-- | Run all components
-- For local development, run each in foreground sequentially.
-- Use tmux or separate terminals for parallel operation.
runAll :: Config -> Sh ()
runAll cfg = do
    echo ":: NativeLink stack components:"
    echo "   1. CAS (content-addressed storage)"
    echo "   2. Scheduler (work distribution)"
    echo "   3. Worker (build execution)"
    echo ""
    echo "For local dev, run each in separate terminals:"
    echo "  nativelink-local cas"
    echo "  nativelink-local scheduler"
    echo "  nativelink-local worker"
    echo ""
    echo "Starting CAS (Ctrl-C to stop, then start next component)..."
    runComponent cfg "cas"

-- | Shell into a container
shellInto :: Config -> Text -> Sh ()
shellInto cfg name = do
    echo $ ":: Shell into " <> name
    case cfgRuntime cfg of
        Firecracker -> runFirecracker cfg name ["/bin/bash"]
        OciNamespace -> runOci cfg name ["/bin/bash"]

main :: IO ()
main = do
    args <- getArgs
    let cfg = parseArgs args
    
    script $ do
        echo "╔══════════════════════════════════════════════════════════════════╗"
        echo "║           NativeLink Local Development                           ║"
        echo "╚══════════════════════════════════════════════════════════════════╝"
        echo ""
        echo $ "Runtime: " <> pack (show $ cfgRuntime cfg)
        echo ""
        
        -- Ensure temp directories exist
        mkdirP "/tmp/nativelink"
        mkdirP "/data" `catch` (\(_ :: SomeException) -> pure ())  -- /data may need root
        
        case cfgComponent cfg of
            Scheduler -> runComponent cfg "scheduler"
            Cas -> runComponent cfg "cas"
            Worker -> runComponent cfg "worker"
            All -> runAll cfg
            Shell name -> shellInto cfg name
        
        echo ""
        echo ":: Done!"
