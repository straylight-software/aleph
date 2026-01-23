{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Deploy NativeLink containers to Fly.io.

Usage: nativelink-deploy [OPTIONS] [COMPONENT]

Components:
  scheduler   Deploy scheduler only
  cas         Deploy CAS only
  worker      Deploy worker only
  push        Push images to registry only
  all         Push and deploy all (default)

Options:
  --registry URL   Container registry (default: ghcr.io/straylight-software/aleph)
  --region REGION  Fly.io region (default: iad)
  --dry-run        Show commands without executing

Example:
  nativelink-deploy              # Deploy all
  nativelink-deploy push         # Just push images
  nativelink-deploy scheduler    # Deploy scheduler only
-}
module Main where

import Aleph.Script
import qualified Data.Text as T
import System.Environment (getArgs)

data Config = Config
    { cfgRegistry :: Text
    , cfgRegion :: Text
    , cfgDryRun :: Bool
    , cfgComponent :: Component
    }

data Component = All | Push | Scheduler | Cas | Worker
    deriving (Eq, Show)

defaultConfig :: Config
defaultConfig = Config
    { cfgRegistry = "ghcr.io/straylight-software/aleph"
    , cfgRegion = "iad"
    , cfgDryRun = False
    , cfgComponent = All
    }

parseArgs :: [String] -> Config
parseArgs = go defaultConfig
  where
    go cfg [] = cfg
    go cfg ("--registry" : r : rest) = go cfg { cfgRegistry = pack r } rest
    go cfg ("--region" : r : rest) = go cfg { cfgRegion = pack r } rest
    go cfg ("--dry-run" : rest) = go cfg { cfgDryRun = True } rest
    go cfg ("scheduler" : rest) = go cfg { cfgComponent = Scheduler } rest
    go cfg ("cas" : rest) = go cfg { cfgComponent = Cas } rest
    go cfg ("worker" : rest) = go cfg { cfgComponent = Worker } rest
    go cfg ("push" : rest) = go cfg { cfgComponent = Push } rest
    go cfg ("all" : rest) = go cfg { cfgComponent = All } rest
    go cfg (_ : rest) = go cfg rest

runCmd :: Config -> [Text] -> Sh ()
runCmd cfg args =
    if cfgDryRun cfg
        then echo $ "[dry-run] " <> T.unwords args
        else run_ (fromText $ Prelude.head args) (Prelude.tail args)

-- | Push container image to registry via nix run
pushImage :: Config -> Text -> Sh ()
pushImage cfg name = do
    echo $ ":: Pushing " <> name <> " to " <> cfgRegistry cfg
    runCmd cfg ["nix", "run", ".#" <> name <> ".copyToGithub"]

-- | Deploy to Fly.io
deployFly :: Config -> Text -> Sh ()
deployFly cfg name = do
    echo $ ":: Deploying " <> name <> " to Fly.io (" <> cfgRegion cfg <> ")"
    
    let appName = "aleph-" <> name
    let image = cfgRegistry cfg <> "/nativelink-" <> name <> ":latest"
    
    -- Deploy using fly deploy with image
    runCmd cfg 
        [ "fly", "deploy"
        , "--app", appName
        , "--image", image
        , "--region", cfgRegion cfg
        , "--remote-only"
        , "--now"
        ]

-- | Ensure CAS volume exists
ensureCasVolume :: Config -> Sh ()
ensureCasVolume cfg = do
    echo ":: Checking CAS volume..."
    volumes <- run "fly" ["volumes", "list", "-a", "aleph-cas", "--json"]
    when (T.null volumes || volumes == "[]") $ do
        echo ":: Creating CAS volume..."
        runCmd cfg ["fly", "volumes", "create", "cas_data", 
                    "--size", "10", 
                    "--region", cfgRegion cfg,
                    "-a", "aleph-cas"]

main :: IO ()
main = do
    args <- getArgs
    let cfg = parseArgs args
    
    script $ do
        echo "╔══════════════════════════════════════════════════════════════════╗"
        echo "║           NativeLink Deployment to Fly.io                        ║"
        echo "╚══════════════════════════════════════════════════════════════════╝"
        echo ""
        
        case cfgComponent cfg of
            Push -> do
                pushImage cfg "nativelink-scheduler"
                pushImage cfg "nativelink-cas"
                pushImage cfg "nativelink-worker"
                
            Scheduler -> do
                pushImage cfg "nativelink-scheduler"
                deployFly cfg "scheduler"
                
            Cas -> do
                pushImage cfg "nativelink-cas"
                ensureCasVolume cfg
                deployFly cfg "cas"
                
            Worker -> do
                pushImage cfg "nativelink-worker"
                deployFly cfg "worker"
                
            All -> do
                -- Push all images first
                pushImage cfg "nativelink-scheduler"
                pushImage cfg "nativelink-cas"
                pushImage cfg "nativelink-worker"
                
                -- Deploy in order: CAS -> Scheduler -> Worker
                ensureCasVolume cfg
                deployFly cfg "cas"
                deployFly cfg "scheduler"
                deployFly cfg "worker"
        
        echo ""
        echo ":: Done!"
