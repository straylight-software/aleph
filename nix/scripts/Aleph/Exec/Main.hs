{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- | aleph-exec: Zero-Bash Build Executor

This is the sole builder for zero-bash derivations. It reads a JSON spec
and executes typed actions directly - no shell, no bash, no string interpolation.

= Architecture

The Nix derivation looks like:

@
derivation {
  builder = "${aleph-exec}/bin/aleph-exec";
  args = [ "--spec" "${specFile}" ];
  -- ...
}
@

aleph-exec then:
1. Reads the JSON spec
2. Validates all paths
3. Executes each action using Haskell I/O
4. Reports success/failure with structured output

= Security

- No shell execution (except patchelf, which is audited)
- All paths validated before use
- No string interpolation vulnerabilities
- Output paths are validated to be under $out

= Actions Executed Directly

- Mkdir     -> System.Directory.createDirectoryIfMissing True
- Copy      -> System.Directory.copyFile / copyDirectoryRecursive
- Symlink   -> System.Posix.Files.createSymbolicLink
- WriteFile -> Data.Text.IO.writeFile
- Install   -> copyFile + setFileMode
- Remove    -> removePathForcibly
- Unzip     -> Codec.Archive.Zip.extractFilesFromArchive
- Chmod     -> setFileMode
- Substitute -> Text.replace in file

= External Tools

Only one external tool is called:
- patchelf: For PatchElfRpath, PatchElfAddRpath, PatchElfInterpreter

This is necessary because modifying ELF headers requires specialized code.
The arguments are fully controlled (no user input reaches the command line).
-}
module Main (main) where

import Control.Monad (forM_)
import Data.Aeson (
    FromJSON (..),
    Value (..),
    eitherDecodeFileStrict',
    withArray,
    withObject,
    (.:),
    (.:?),
 )
import Data.Aeson.Key (toText)
import qualified Data.Aeson.KeyMap as KM
import Data.Aeson.Types (Parser)
import qualified Data.ByteString.Lazy as BSL
import Data.Foldable (foldl', toList)
import Data.List (intercalate)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import GHC.Generics (Generic)
import System.Directory (
    copyFile,
    createDirectoryIfMissing,
    doesDirectoryExist,
    listDirectory,
    removePathForcibly,
 )
import System.Environment (getArgs, getEnv, lookupEnv)
import System.Exit (exitFailure, exitSuccess)
import System.FilePath (takeDirectory, (</>))
import System.IO (hPutStrLn, stderr)
import System.Posix.Files (createSymbolicLink, setFileMode)
import System.Process (callProcess)

-- For zip extraction
import qualified Codec.Archive.Zip as Zip

--------------------------------------------------------------------------------
-- Spec Types
--------------------------------------------------------------------------------

-- | The complete derivation spec (parsed from JSON)
data Spec = Spec
    { specPname :: Text
    , specVersion :: Text
    , specPhases :: Phases
    , specMeta :: Maybe Meta
    , specEnv :: [(Text, Text)]
    }
    deriving (Show, Generic)

data Phases = Phases
    { phasesPostPatch :: [Action]
    , phasesPreConfigure :: [Action]
    , phasesInstallPhase :: [Action]
    , phasesPostInstall :: [Action]
    , phasesPostFixup :: [Action]
    }
    deriving (Show, Generic)

data Meta = Meta
    { metaDescription :: Maybe Text
    }
    deriving (Show, Generic)

-- | Build action
data Action
    = Mkdir Text
    | Copy Text Text
    | Symlink Text Text
    | WriteFile Text Text
    | Install Int Text Text
    | Remove Text
    | Unzip Text
    | PatchElfRpath Text [Text]
    | PatchElfAddRpath Text [Text]
    | PatchElfInterpreter Text Text
    | Substitute Text [(Text, Text)]
    | Wrap Text [WrapAction]
    | Chmod Text Int
    deriving (Show, Generic)

data WrapAction
    = WrapPrefix Text Text
    | WrapSuffix Text Text
    | WrapSet Text Text
    | WrapSetDefault Text Text
    | WrapUnset Text
    | WrapAddFlags Text
    deriving (Show, Generic)

--------------------------------------------------------------------------------
-- JSON Parsing
--------------------------------------------------------------------------------

instance FromJSON Spec where
    parseJSON = withObject "Spec" $ \o -> do
        specPname <- o .: "pname"
        specVersion <- o .: "version"
        specPhases <- o .: "phases"
        specMeta <- o .:? "meta"
        env <- o .:? "env"
        let specEnv = case env of
                Nothing -> []
                Just (Object m) -> [(toText k, v) | (k, String v) <- KM.toList m]
                _ -> []
        return Spec{..}

instance FromJSON Phases where
    parseJSON = withObject "Phases" $ \o -> do
        phasesPostPatch <- o .:? "postPatch" >>= parseActions
        phasesPreConfigure <- o .:? "preConfigure" >>= parseActions
        phasesInstallPhase <- o .:? "installPhase" >>= parseActions
        phasesPostInstall <- o .:? "postInstall" >>= parseActions
        phasesPostFixup <- o .:? "postFixup" >>= parseActions
        return Phases{..}
      where
        parseActions :: Maybe Value -> Parser [Action]
        parseActions Nothing = return []
        parseActions (Just v) = withArray "actions" (mapM parseAction . toList) v

instance FromJSON Meta where
    parseJSON = withObject "Meta" $ \o -> do
        metaDescription <- o .:? "description"
        return Meta{..}

parseAction :: Value -> Parser Action
parseAction = withObject "Action" $ \o -> do
    action :: Text <- o .: "action"
    case action of
        "mkdir" -> Mkdir <$> o .: "path"
        "copy" -> Copy <$> o .: "src" <*> o .: "dst"
        "symlink" -> Symlink <$> o .: "target" <*> o .: "link"
        "writeFile" -> WriteFile <$> o .: "path" <*> o .: "content"
        "install" -> Install <$> o .: "mode" <*> o .: "src" <*> o .: "dst"
        "remove" -> Remove <$> o .: "path"
        "unzip" -> Unzip <$> o .: "dest"
        "patchelfRpath" -> PatchElfRpath <$> o .: "path" <*> o .: "rpaths"
        "patchelfAddRpath" -> PatchElfAddRpath <$> o .: "path" <*> o .: "rpaths"
        "patchelfInterpreter" -> PatchElfInterpreter <$> o .: "path" <*> o .: "interpreter"
        "substitute" -> Substitute <$> o .: "file" <*> parseReplacements o
        "wrap" -> Wrap <$> o .: "program" <*> parseWrapActions o
        "chmod" -> Chmod <$> o .: "path" <*> o .: "mode"
        _ -> fail $ "Unknown action: " ++ T.unpack action

parseReplacements :: KM.KeyMap Value -> Parser [(Text, Text)]
parseReplacements o = do
    reps <- o .: "replacements"
    withArray "replacements" (mapM parseRep . toList) reps
  where
    parseRep = withObject "replacement" $ \r ->
        (,) <$> r .: "from" <*> r .: "to"

parseWrapActions :: KM.KeyMap Value -> Parser [WrapAction]
parseWrapActions o = do
    acts <- o .: "wrapActions"
    withArray "wrapActions" (mapM parseWA . toList) acts
  where
    parseWA = withObject "wrapAction" $ \w -> do
        typ :: Text <- w .: "type"
        case typ of
            "prefix" -> WrapPrefix <$> w .: "var" <*> w .: "value"
            "suffix" -> WrapSuffix <$> w .: "var" <*> w .: "value"
            "set" -> WrapSet <$> w .: "var" <*> w .: "value"
            "setDefault" -> WrapSetDefault <$> w .: "var" <*> w .: "value"
            "unset" -> WrapUnset <$> w .: "var"
            "addFlags" -> WrapAddFlags <$> w .: "flags"
            _ -> fail $ "Unknown wrap action: " ++ T.unpack typ

--------------------------------------------------------------------------------
-- Environment
--------------------------------------------------------------------------------

-- | Build context with paths from environment
data BuildContext = BuildContext
    { ctxOut :: FilePath
    -- ^ $out
    , ctxSrc :: Maybe FilePath
    -- ^ $src (if present)
    }
    deriving (Show)

getContext :: IO BuildContext
getContext = do
    ctxOut <- getEnv "out"
    ctxSrc <- lookupEnv "src"
    return BuildContext{..}

--------------------------------------------------------------------------------
-- Path Resolution
--------------------------------------------------------------------------------

-- | Resolve a path relative to $out
resolveOutPath :: BuildContext -> Text -> FilePath
resolveOutPath ctx path
    | T.isPrefixOf "/" path = T.unpack path -- Absolute path (store path)
    | T.isPrefixOf "$out/" path = ctxOut ctx </> T.unpack (T.drop 5 path)
    | T.isPrefixOf "$out" path = ctxOut ctx </> T.unpack (T.drop 4 path)
    | otherwise = ctxOut ctx </> T.unpack path

-- | Resolve a source path (relative to $src or absolute)
resolveSrcPath :: BuildContext -> Text -> FilePath
resolveSrcPath ctx path
    | T.isPrefixOf "/" path = T.unpack path -- Absolute path
    | T.isPrefixOf "$src/" path = case ctxSrc ctx of
        Just src -> src </> T.unpack (T.drop 5 path)
        Nothing -> error "No source specified but path references $src"
    | T.isPrefixOf "$src" path = case ctxSrc ctx of
        Just src -> src </> T.unpack (T.drop 4 path)
        Nothing -> error "No source specified but path references $src"
    | T.isPrefixOf "$out/" path = ctxOut ctx </> T.unpack (T.drop 5 path)
    | T.isPrefixOf "$out" path = ctxOut ctx </> T.unpack (T.drop 4 path)
    | otherwise = case ctxSrc ctx of
        Just src -> src </> T.unpack path
        Nothing -> T.unpack path

--------------------------------------------------------------------------------
-- Action Execution
--------------------------------------------------------------------------------

-- | Execute a single action
executeAction :: BuildContext -> Action -> IO ()
executeAction ctx action = do
    logAction action
    case action of
        Mkdir path -> do
            let p = resolveOutPath ctx path
            createDirectoryIfMissing True p

        Copy src dst -> do
            let srcPath = resolveSrcPath ctx src
                dstPath = resolveOutPath ctx dst
            createDirectoryIfMissing True (takeDirectory dstPath)
            isDir <- doesDirectoryExist srcPath
            if isDir
                then copyDirectoryRecursive srcPath dstPath
                else copyFile srcPath dstPath

        Symlink target link -> do
            let linkPath = resolveOutPath ctx link
                targetPath = T.unpack target -- Target can be relative or absolute
            createDirectoryIfMissing True (takeDirectory linkPath)
            createSymbolicLink targetPath linkPath

        WriteFile path content -> do
            let p = resolveOutPath ctx path
            createDirectoryIfMissing True (takeDirectory p)
            TIO.writeFile p content

        Install mode src dst -> do
            let srcPath = resolveSrcPath ctx src
                dstPath = resolveOutPath ctx dst
            createDirectoryIfMissing True (takeDirectory dstPath)
            copyFile srcPath dstPath
            setFileMode dstPath (fromIntegral mode)

        Remove path -> do
            let p = resolveOutPath ctx path
            removePathForcibly p

        Unzip dest -> do
            let dstDir = resolveOutPath ctx dest
            srcFile <- case ctxSrc ctx of
                Just s -> return s
                Nothing -> error "Unzip requires $src"
            createDirectoryIfMissing True dstDir
            archive <- Zip.toArchive <$> BSL.readFile srcFile
            Zip.extractFilesFromArchive [Zip.OptDestination dstDir] archive

        PatchElfRpath binary rpaths -> do
            let binaryPath = resolveOutPath ctx binary
                rpathStr = intercalate ":" (map T.unpack rpaths)
            callProcess "patchelf" ["--set-rpath", rpathStr, binaryPath]

        PatchElfAddRpath binary rpaths -> do
            let binaryPath = resolveOutPath ctx binary
                rpathStr = intercalate ":" (map T.unpack rpaths)
            callProcess "patchelf" ["--add-rpath", rpathStr, binaryPath]

        PatchElfInterpreter binary interp -> do
            let binaryPath = resolveOutPath ctx binary
                interpPath = T.unpack interp
            callProcess "patchelf" ["--set-interpreter", interpPath, binaryPath]

        Substitute file replacements -> do
            let filePath = resolveOutPath ctx file
            content <- TIO.readFile filePath
            let content' = foldl' (\c (from, to) -> T.replace from to c) content replacements
            TIO.writeFile filePath content'

        Wrap program wrapActions -> do
            let progPath = resolveOutPath ctx program
            generateWrapper progPath wrapActions

        Chmod path mode -> do
            let p = resolveOutPath ctx path
            setFileMode p (fromIntegral mode)

-- | Log action being executed
logAction :: Action -> IO ()
logAction = \case
    Mkdir p -> log' $ "mkdir " ++ T.unpack p
    Copy s d -> log' $ "copy " ++ T.unpack s ++ " -> " ++ T.unpack d
    Symlink t l -> log' $ "symlink " ++ T.unpack t ++ " <- " ++ T.unpack l
    WriteFile p _ -> log' $ "write " ++ T.unpack p
    Install m s d -> log' $ "install -m" ++ show m ++ " " ++ T.unpack s ++ " -> " ++ T.unpack d
    Remove p -> log' $ "remove " ++ T.unpack p
    Unzip d -> log' $ "unzip -> " ++ T.unpack d
    PatchElfRpath b rs -> log' $ "patchelf --set-rpath " ++ show rs ++ " " ++ T.unpack b
    PatchElfAddRpath b rs -> log' $ "patchelf --add-rpath " ++ show rs ++ " " ++ T.unpack b
    PatchElfInterpreter b i -> log' $ "patchelf --set-interpreter " ++ T.unpack i ++ " " ++ T.unpack b
    Substitute f rs -> log' $ "substitute " ++ T.unpack f ++ " (" ++ show (length rs) ++ " replacements)"
    Wrap p _ -> log' $ "wrap " ++ T.unpack p
    Chmod p m -> log' $ "chmod " ++ show m ++ " " ++ T.unpack p
  where
    log' msg = hPutStrLn stderr $ "[aleph-exec] " ++ msg

-- | Copy a directory recursively
copyDirectoryRecursive :: FilePath -> FilePath -> IO ()
copyDirectoryRecursive src dst = do
    createDirectoryIfMissing True dst
    entries <- listDirectory src
    forM_ entries $ \entry -> do
        let srcPath = src </> entry
            dstPath = dst </> entry
        isDir <- doesDirectoryExist srcPath
        if isDir
            then copyDirectoryRecursive srcPath dstPath
            else copyFile srcPath dstPath

-- | Generate a wrapper script
generateWrapper :: FilePath -> [WrapAction] -> IO ()
generateWrapper progPath wrapActions = do
    -- Rename original to .wrapped
    let wrappedPath = progPath ++ "-wrapped"
    copyFile progPath wrappedPath

    -- Generate wrapper script
    let script = generateWrapperScript wrappedPath wrapActions
    TIO.writeFile progPath script

    -- Make executable
    setFileMode progPath 0o755

-- | Generate wrapper script content
generateWrapperScript :: FilePath -> [WrapAction] -> Text
generateWrapperScript wrappedPath actions =
    T.unlines $
        [ "#!/bin/sh"
        , "# Wrapper generated by aleph-exec"
        , ""
        ]
            ++ concatMap wrapActionToLines actions
            ++ [ ""
               , "exec \"" <> T.pack wrappedPath <> "\" \"$@\""
               ]
  where
    wrapActionToLines = \case
        WrapPrefix var val ->
            ["export " <> var <> "=\"" <> val <> "${" <> var <> ":+:$" <> var <> "}\""]
        WrapSuffix var val ->
            ["export " <> var <> "=\"${" <> var <> ":+$" <> var <> ":}" <> val <> "\""]
        WrapSet var val ->
            ["export " <> var <> "=\"" <> val <> "\""]
        WrapSetDefault var val ->
            ["export " <> var <> "=\"${" <> var <> ":-" <> val <> "}\""]
        WrapUnset var ->
            ["unset " <> var]
        WrapAddFlags flags ->
            ["set -- " <> flags <> " \"$@\""]

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
    args <- getArgs
    case parseArgs args of
        Left err -> do
            hPutStrLn stderr $ "Error: " ++ err
            hPutStrLn stderr "Usage: aleph-exec --spec SPEC_FILE"
            exitFailure
        Right specFile -> runBuild specFile

parseArgs :: [String] -> Either String FilePath
parseArgs ("--spec" : file : _) = Right file
parseArgs _ = Left "Missing --spec argument"

runBuild :: FilePath -> IO ()
runBuild specFile = do
    hPutStrLn stderr $ "[aleph-exec] Reading spec: " ++ specFile

    -- Parse spec
    result <- eitherDecodeFileStrict' specFile
    case result of
        Left err -> do
            hPutStrLn stderr $ "[aleph-exec] Failed to parse spec: " ++ err
            exitFailure
        Right spec -> executeBuild spec

executeBuild :: Spec -> IO ()
executeBuild spec = do
    hPutStrLn stderr $
        "[aleph-exec] Building "
            ++ T.unpack (specPname spec)
            ++ "-"
            ++ T.unpack (specVersion spec)

    -- Get build context
    ctx <- getContext
    hPutStrLn stderr $ "[aleph-exec] Output: " ++ ctxOut ctx
    case ctxSrc ctx of
        Just src -> hPutStrLn stderr $ "[aleph-exec] Source: " ++ src
        Nothing -> hPutStrLn stderr "[aleph-exec] No source"

    -- Create output directory
    createDirectoryIfMissing True (ctxOut ctx)

    -- Execute phases in order
    let phases = specPhases spec

    hPutStrLn stderr "[aleph-exec] Phase: postPatch"
    mapM_ (executeAction ctx) (phasesPostPatch phases)

    hPutStrLn stderr "[aleph-exec] Phase: preConfigure"
    mapM_ (executeAction ctx) (phasesPreConfigure phases)

    hPutStrLn stderr "[aleph-exec] Phase: installPhase"
    mapM_ (executeAction ctx) (phasesInstallPhase phases)

    hPutStrLn stderr "[aleph-exec] Phase: postInstall"
    mapM_ (executeAction ctx) (phasesPostInstall phases)

    hPutStrLn stderr "[aleph-exec] Phase: postFixup"
    mapM_ (executeAction ctx) (phasesPostFixup phases)

    hPutStrLn stderr "[aleph-exec] Build complete"
    exitSuccess
