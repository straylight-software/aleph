# Write a Script

Create a Haskell script with Aleph.Script.

## Basic Script

Create `my-script.hs`:

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script

main :: IO ()
main = script $ do
    echo "Starting..."
    
    -- List files
    files <- ls "."
    echo $ format ("Found "%d%" files") (length files)
    
    -- Filter and process
    let nixFiles = filter (hasExtension "nix") files
    for_ nixFiles $ \f -> do
        content <- readFile f
        echo $ format ("File "%fp%" has "%d%" lines") f (length (lines content))
    
    echo "Done!"
```

## Package in Nix

```nix
perSystem = { config, pkgs, ... }:
  let
    inherit (config.aleph.prelude) ghc;
  in {
    packages.my-script = ghc.turtle-script {
      name = "my-script";
      src = ./my-script.hs;
      deps = [ ];           # Runtime dependencies
      hs-deps = p: [ ];     # Additional Haskell packages
    };
  };
```

## With Runtime Dependencies

```nix
packages.container-script = ghc.turtle-script {
  name = "container-script";
  src = ./container-script.hs;
  deps = [
    pkgs.bubblewrap
    pkgs.crane
    pkgs.jq
  ];
  hs-deps = p: [
    p.aeson
    p.optparse-applicative
  ];
};
```

## Script Examples

### File Processing

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script

main :: IO ()
main = script $ do
    args <- liftIO getArgs
    case args of
        [dir] -> processDir (fromText (pack dir))
        _ -> die "usage: process-files <dir>"

processDir :: FilePath -> Sh ()
processDir dir = do
    files <- ls dir
    for_ files $ \f -> do
        isDir <- test_d f
        if isDir
            then processDir f
            else processFile f

processFile :: FilePath -> Sh ()
processFile f = do
    let ext = extension f
    case ext of
        Just "nix" -> echo $ format ("Nix: "%fp) f
        Just "hs" -> echo $ format ("Haskell: "%fp) f
        _ -> pure ()
```

### Running Commands

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script

main :: IO ()
main = script $ do
    -- Capture output
    gitStatus <- run "git" ["status", "--porcelain"]
    unless (T.null (strip gitStatus)) $
        echo "Working directory dirty!"
    
    -- Run without capturing
    run_ "nix" ["build", ".#default"]
    
    -- Run bash command
    result <- bash "echo $HOME"
    echo $ "Home: " <> strip result
```

### Error Handling

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script
import Control.Exception (try, SomeException)

main :: IO ()
main = script $ do
    result <- errExit False $ run "might-fail" []
    code <- exitCode
    
    if code == 0
        then echo $ "Success: " <> result
        else echoErr "Command failed, continuing..."
    
    -- Or use try for exceptions
    maybeResult <- liftIO $ try $ shelly $ run "other-cmd" []
    case maybeResult of
        Left (e :: SomeException) -> echoErr $ "Error: " <> pack (show e)
        Right output -> echo output
```

### Argument Parsing

```haskell
{-# LANGUAGE OverloadedStrings #-}
import Aleph.Script
import Options.Applicative

data Options = Options
    { optVerbose :: Bool
    , optOutput :: FilePath
    , optInputs :: [FilePath]
    }

optParser :: Parser Options
optParser = Options
    <$> switch (long "verbose" <> short 'v' <> help "Verbose output")
    <*> option (fromText . pack <$> str) 
        (long "output" <> short 'o' <> metavar "FILE" <> value "out")
    <*> many (argument (fromText . pack <$> str) (metavar "INPUT..."))

main :: IO ()
main = do
    opts <- execParser (info (optParser <**> helper) fullDesc)
    shelly $ runWithOpts opts

runWithOpts :: Options -> Sh ()
runWithOpts opts = do
    when (optVerbose opts) $
        echo "Verbose mode enabled"
    
    for_ (optInputs opts) $ \input ->
        echo $ format ("Processing: "%fp) input
```

## Development Workflow

```bash
# Enter script development shell
nix develop .#aleph-script

# Run script directly (interpreted, ~160ms startup)
runghc -i. my-script.hs

# Or compile for fast startup (~2ms)
ghc -O2 my-script.hs -o my-script
./my-script
```

## Pre-compiled Scripts

Use the pre-compiled scripts in the overlay:

```nix
{
  # Add to devshell
  devShells.default = pkgs.mkShell {
    packages = [
      pkgs.aleph.script.compiled.oci-run
      pkgs.aleph.script.compiled.fhs-run
      pkgs.aleph.script.compiled.gpu-run
    ];
  };
}
```
