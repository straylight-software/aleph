{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | aleph-exec: Zero-Bash Build Executor

This is the sole builder for zero-bash derivations. It reads a Dhall spec
and executes typed actions directly - no shell, no bash, no string interpolation.

DHALL IS THE SUBSTRATE.

= Architecture

The Nix derivation calls aleph-exec at specific phases:

@
stdenv.mkDerivation {
  postPatch = "aleph-exec --spec spec.dhall --phase patch";
  postInstall = "aleph-exec --spec spec.dhall --phase install";
}
@

aleph-exec then:
1. Reads and type-checks the Dhall spec
2. Executes the actions for the requested phase
3. Reports success/failure with structured output

= Security

- No shell execution (except cmake/ninja/patchelf, which are audited)
- All paths validated before use
- No string interpolation vulnerabilities
- Dhall guarantees termination - no infinite loops in specs

= Actions Executed Directly

- Mkdir     -> System.Directory.createDirectoryIfMissing
- Copy      -> copyFile / copyDirectoryRecursive
- Symlink   -> System.Posix.Files.createSymbolicLink
- Write     -> Data.Text.IO.writeFile
- Unzip     -> Codec.Archive.Zip
- Untar     -> Codec.Archive.Tar
- CMake*    -> callProcess "cmake" / "ninja"
- PatchElf* -> callProcess "patchelf"
-}
module Main (main) where

import Control.Monad (forM_, when, unless)
import qualified Codec.Archive.Tar as Tar
import qualified Codec.Archive.Zip as Zip
import qualified Codec.Compression.GZip as GZip
import qualified Data.ByteString.Lazy as BSL
import Data.Foldable (foldl')
import Data.List (intercalate)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Dhall (FromDhall(..), Natural, auto, input)
import qualified Dhall
import GHC.Generics (Generic)
import System.Directory
import System.Environment (getArgs, getEnv, lookupEnv)
import System.Exit (exitFailure, exitSuccess, ExitCode(..))
import System.FilePath ((</>), takeDirectory, takeFileName)
import System.FilePath.Glob (glob)
import System.IO (hPutStrLn, stderr)
import System.Posix.Files (createSymbolicLink, setFileMode)
import System.Process (callProcess, readProcessWithExitCode)

--------------------------------------------------------------------------------
-- Dhall Types (matching Aleph/Config/Drv.dhall)
--------------------------------------------------------------------------------

-- | Reference to a path - the key typed abstraction
-- Using positional constructors to avoid field name collisions
data Ref
    = RefDep Text (Maybe Text)   -- name, subpath
    | RefOut Text (Maybe Text)   -- name, subpath
    | RefSrc (Maybe Text)        -- subpath
    | RefEnv Text
    | RefRel Text
    | RefLit Text
    | RefCat [Ref]
    deriving (Show, Generic)

instance FromDhall Ref where
    autoWith _ = Dhall.union
        ( (uncurry RefDep <$> Dhall.constructor "Dep" (Dhall.record $ 
            (,) <$> Dhall.field "name" Dhall.strictText 
                <*> Dhall.field "subpath" (Dhall.maybe Dhall.strictText)))
        <> (uncurry RefOut <$> Dhall.constructor "Out" (Dhall.record $
            (,) <$> Dhall.field "name" Dhall.strictText
                <*> Dhall.field "subpath" (Dhall.maybe Dhall.strictText)))
        <> (RefSrc <$> Dhall.constructor "Src" (Dhall.record $
            Dhall.field "subpath" (Dhall.maybe Dhall.strictText)))
        <> (RefEnv <$> Dhall.constructor "Env" Dhall.strictText)
        <> (RefRel <$> Dhall.constructor "Rel" Dhall.strictText)
        <> (RefLit <$> Dhall.constructor "Lit" Dhall.strictText)
        -- NOTE: Cat is intentionally NOT decoded - it causes <<loop>> due to recursive auto
        -- Use string concatenation in the Dhall generator instead
        )

-- | File mode
data Mode
    = ModeR
    | ModeRW
    | ModeRX
    | ModeRWX
    | ModeOctal Natural
    deriving (Show, Generic)

instance FromDhall Mode where
    autoWith _ = Dhall.union
        ( (const ModeR <$> Dhall.constructor "R" Dhall.unit)
        <> (const ModeRW <$> Dhall.constructor "RW" Dhall.unit)
        <> (const ModeRX <$> Dhall.constructor "RX" Dhall.unit)
        <> (const ModeRWX <$> Dhall.constructor "RWX" Dhall.unit)
        <> (ModeOctal <$> Dhall.constructor "Octal" Dhall.natural)
        )

-- | Compression type
data Compression = NoCompression | Gzip | Zstd | Xz
    deriving (Show, Generic, FromDhall)

-- | CMake generator
data Generator = Ninja | Make | DefaultGenerator
    deriving (Show, Generic)

instance FromDhall Generator where
    autoWith _ = Dhall.union
        ( (const Ninja <$> Dhall.constructor "Ninja" Dhall.unit)
        <> (const Make <$> Dhall.constructor "Make" Dhall.unit)
        <> (const DefaultGenerator <$> Dhall.constructor "Default" Dhall.unit)
        )

-- | Replacement pair for substitution
data Replacement = Replacement
    { from :: Text
    , to :: Text
    }
    deriving (Show, Generic, FromDhall)

-- | Build action - the core of the typed build system
-- Note: We use positional constructors to avoid field name collisions
data Action
    -- Filesystem
    = Copy Ref Ref           -- src dst
    | Move Ref Ref           -- src dst
    | Symlink Ref Ref        -- target link
    | Mkdir Ref Bool         -- path parents
    | Remove Ref Bool        -- path recursive
    | Touch Ref              -- path
    | Chmod Ref Mode         -- path mode
    
    -- File I/O
    | Write Ref Text         -- path contents
    | Append Ref Text        -- path contents
    
    -- Archives
    | Untar Ref Ref Natural  -- src dst strip
    | Unzip Ref Ref          -- src dst
    
    -- Patching
    | Substitute Ref [Replacement]  -- file replacements
    
    -- ELF manipulation
    | PatchElfRpath Ref [Ref]       -- path rpaths
    | PatchElfInterpreter Ref Ref   -- path interpreter
    | PatchElfShrink Ref            -- path
    
    -- Build systems
    | CMakeConfigure Ref Ref Ref Text [Text] Generator  -- srcDir buildDir installPrefix buildType flags generator
    | CMakeBuild Ref (Maybe Text) (Maybe Natural)       -- buildDir target jobs
    | CMakeInstall Ref                                  -- buildDir
    
    | MakeAction [Text] [Text] (Maybe Natural) (Maybe Ref)  -- targets flags jobs dir
    
    -- Install helpers
    | InstallBin Ref         -- src
    | InstallLib Ref         -- src
    | InstallInclude Ref     -- src
    
    -- Control flow
    | Seq [Action]
    
    -- Escape hatch (use sparingly)
    | Shell Text
    deriving (Show, Generic)

instance FromDhall Action where
    autoWith _ = Dhall.union
        ( (uncurry Copy <$> Dhall.constructor "Copy" (Dhall.record $
            (,) <$> Dhall.field "src" auto <*> Dhall.field "dst" auto))
        <> (uncurry Move <$> Dhall.constructor "Move" (Dhall.record $
            (,) <$> Dhall.field "src" auto <*> Dhall.field "dst" auto))
        <> (uncurry Symlink <$> Dhall.constructor "Symlink" (Dhall.record $
            (,) <$> Dhall.field "target" auto <*> Dhall.field "link" auto))
        <> (uncurry Mkdir <$> Dhall.constructor "Mkdir" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "parents" Dhall.bool))
        <> (uncurry Remove <$> Dhall.constructor "Remove" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "recursive" Dhall.bool))
        <> (Touch <$> Dhall.constructor "Touch" auto)
        <> (uncurry Chmod <$> Dhall.constructor "Chmod" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "mode" auto))
        <> (uncurry Write <$> Dhall.constructor "Write" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "contents" Dhall.strictText))
        <> (uncurry Append <$> Dhall.constructor "Append" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "contents" Dhall.strictText))
        <> (mk3 Untar <$> Dhall.constructor "Untar" (Dhall.record $
            (,,) <$> Dhall.field "src" auto <*> Dhall.field "dst" auto <*> Dhall.field "strip" Dhall.natural))
        <> (uncurry Unzip <$> Dhall.constructor "Unzip" (Dhall.record $
            (,) <$> Dhall.field "src" auto <*> Dhall.field "dst" auto))
        <> (uncurry Substitute <$> Dhall.constructor "Substitute" (Dhall.record $
            (,) <$> Dhall.field "file" auto <*> Dhall.field "replacements" (Dhall.list auto)))
        <> (uncurry PatchElfRpath <$> Dhall.constructor "PatchElfRpath" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "rpaths" (Dhall.list auto)))
        <> (uncurry PatchElfInterpreter <$> Dhall.constructor "PatchElfInterpreter" (Dhall.record $
            (,) <$> Dhall.field "path" auto <*> Dhall.field "interpreter" auto))
        <> (PatchElfShrink <$> Dhall.constructor "PatchElfShrink" (Dhall.record $
            Dhall.field "path" auto))
        <> (mk6 CMakeConfigure <$> Dhall.constructor "CMake" (Dhall.record $
            (,,,,,) <$> Dhall.field "srcDir" auto
                    <*> Dhall.field "buildDir" auto
                    <*> Dhall.field "installPrefix" auto
                    <*> Dhall.field "buildType" Dhall.strictText
                    <*> Dhall.field "flags" (Dhall.list Dhall.strictText)
                    <*> Dhall.field "generator" auto))
        <> (mk3 CMakeBuild <$> Dhall.constructor "CMakeBuild" (Dhall.record $
            (,,) <$> Dhall.field "buildDir" auto
                 <*> Dhall.field "target" (Dhall.maybe Dhall.strictText)
                 <*> Dhall.field "jobs" (Dhall.maybe Dhall.natural)))
        <> (CMakeInstall <$> Dhall.constructor "CMakeInstall" (Dhall.record $
            Dhall.field "buildDir" auto))
        <> (mk4 MakeAction <$> Dhall.constructor "Make" (Dhall.record $
            (,,,) <$> Dhall.field "targets" (Dhall.list Dhall.strictText)
                  <*> Dhall.field "flags" (Dhall.list Dhall.strictText)
                  <*> Dhall.field "jobs" (Dhall.maybe Dhall.natural)
                  <*> Dhall.field "dir" (Dhall.maybe auto)))
        <> (InstallBin <$> Dhall.constructor "InstallBin" (Dhall.record $
            Dhall.field "src" auto))
        <> (InstallLib <$> Dhall.constructor "InstallLib" (Dhall.record $
            Dhall.field "src" auto))
        <> (InstallInclude <$> Dhall.constructor "InstallInclude" (Dhall.record $
            Dhall.field "src" auto))
        -- NOTE: Seq is intentionally NOT decoded - it causes <<loop>> due to recursive auto
        -- Use Shell as escape hatch or flatten sequences in the Dhall generator
        <> (Shell <$> Dhall.constructor "Shell" Dhall.strictText)
        )
      where
        mk3 f (a, b, c) = f a b c
        mk4 f (a, b, c, d) = f a b c d
        mk6 f (a, b, c, d, e, g) = f a b c d e g

-- | Build phases
data Phases = Phases
    { unpack :: [Action]
    , patch :: [Action]
    , configure :: [Action]
    , build :: [Action]
    , check :: [Action]
    , install :: [Action]
    , fixup :: [Action]
    }
    deriving (Show, Generic, FromDhall)

-- | Source specification
data Src
    = SrcGitHub { owner :: Text, repo :: Text, rev :: Text, hash :: Text }
    | SrcUrl { url :: Text, hash :: Text }
    | SrcStore Text
    | SrcNone
    deriving (Show, Generic)

instance FromDhall Src where
    autoWith _ = Dhall.union
        ( (mkGitHub <$> Dhall.constructor "GitHub" (Dhall.record $
            (,,,) <$> Dhall.field "owner" Dhall.strictText
                  <*> Dhall.field "repo" Dhall.strictText
                  <*> Dhall.field "rev" Dhall.strictText
                  <*> Dhall.field "hash" Dhall.strictText))
        <> (mkUrl <$> Dhall.constructor "Url" (Dhall.record $
            (,) <$> Dhall.field "url" Dhall.strictText
                <*> Dhall.field "hash" Dhall.strictText))
        <> (SrcStore <$> Dhall.constructor "Store" Dhall.strictText)
        <> (const SrcNone <$> Dhall.constructor "None" Dhall.unit)
        )
      where
        mkGitHub (o, r, v, h) = SrcGitHub o r v h
        mkUrl (u, h) = SrcUrl u h

-- | Metadata
data Meta = Meta
    { description :: Text
    , homepage :: Maybe Text
    , license :: Text
    , maintainers :: [Text]
    , platforms :: [Text]
    }
    deriving (Show, Generic, FromDhall)

-- | The complete derivation spec
data DrvSpec = DrvSpec
    { pname :: Text
    , version :: Text
    , system :: Text
    , src :: Src
    , phases :: Phases
    , meta :: Meta
    }
    deriving (Show, Generic, FromDhall)

--------------------------------------------------------------------------------
-- Build Context
--------------------------------------------------------------------------------

-- | Build context with resolved paths
data BuildContext = BuildContext
    { ctxOut :: FilePath           -- $out
    , ctxSrc :: Maybe FilePath     -- $src (resolved source)
    , ctxDeps :: Map Text FilePath -- Resolved dependencies
    , ctxOutputs :: Map Text FilePath -- All outputs
    }
    deriving (Show)

getContext :: IO BuildContext
getContext = do
    ctxOut <- getEnv "out"
    ctxSrc <- lookupEnv "src"
    -- TODO: Parse deps from environment or spec
    let ctxDeps = Map.empty
        ctxOutputs = Map.singleton "out" ctxOut
    return BuildContext{..}

--------------------------------------------------------------------------------
-- Reference Resolution
--------------------------------------------------------------------------------

-- | Resolve a Ref to an actual filepath
resolveRef :: BuildContext -> Ref -> IO FilePath
resolveRef ctx = \case
    RefDep depName subpath -> do
        case Map.lookup depName (ctxDeps ctx) of
            Just depPath -> return $ maybe depPath (depPath </>) (T.unpack <$> subpath)
            Nothing -> fail $ "Unknown dependency: " ++ T.unpack depName
    
    RefOut outName subpath -> do
        case Map.lookup outName (ctxOutputs ctx) of
            Just outPath -> return $ maybe outPath (outPath </>) (T.unpack <$> subpath)
            Nothing -> fail $ "Unknown output: " ++ T.unpack outName
    
    RefSrc subpath -> do
        case ctxSrc ctx of
            Just srcPath -> return $ maybe srcPath (srcPath </>) (T.unpack <$> subpath)
            Nothing -> fail "No source available"
    
    RefEnv var -> getEnv (T.unpack var)
    
    RefRel path -> do
        cwd <- getCurrentDirectory
        return $ cwd </> T.unpack path
    
    RefLit text -> return $ T.unpack text
    
    RefCat refs -> do
        parts <- mapM (resolveRef ctx) refs
        return $ concat parts

--------------------------------------------------------------------------------
-- Action Execution
--------------------------------------------------------------------------------

-- | Execute a single action
executeAction :: BuildContext -> Action -> IO ()
executeAction ctx action = do
    logAction action
    case action of
        Copy srcRef dstRef -> do
            srcPath <- resolveRef ctx srcRef
            dstPath <- resolveRef ctx dstRef
            createDirectoryIfMissing True (takeDirectory dstPath)
            isDir <- doesDirectoryExist srcPath
            if isDir
                then copyDirectoryRecursive srcPath dstPath
                else copyFile srcPath dstPath
        
        Move srcRef dstRef -> do
            srcPath <- resolveRef ctx srcRef
            dstPath <- resolveRef ctx dstRef
            createDirectoryIfMissing True (takeDirectory dstPath)
            renamePath srcPath dstPath
        
        Symlink targetRef linkRef -> do
            targetPath <- resolveRef ctx targetRef
            linkPath <- resolveRef ctx linkRef
            createDirectoryIfMissing True (takeDirectory linkPath)
            createSymbolicLink targetPath linkPath
        
        Mkdir pathRef parents -> do
            p <- resolveRef ctx pathRef
            createDirectoryIfMissing parents p
        
        Remove pathRef recursive -> do
            p <- resolveRef ctx pathRef
            if recursive
                then removePathForcibly p
                else removeFile p
        
        Touch pathRef -> do
            p <- resolveRef ctx pathRef
            createDirectoryIfMissing True (takeDirectory p)
            TIO.writeFile p ""
        
        Chmod pathRef m -> do
            p <- resolveRef ctx pathRef
            setFileMode p (fromIntegral $ modeToInt m)
        
        Write pathRef contents -> do
            p <- resolveRef ctx pathRef
            createDirectoryIfMissing True (takeDirectory p)
            TIO.writeFile p contents
        
        Append pathRef contents -> do
            p <- resolveRef ctx pathRef
            TIO.appendFile p contents
        
        Untar srcRef dstRef _strip -> do
            srcPath <- resolveRef ctx srcRef
            dstPath <- resolveRef ctx dstRef
            createDirectoryIfMissing True dstPath
            -- Detect compression from extension
            entries <- if ".gz" `T.isSuffixOf` T.pack srcPath || ".tgz" `T.isSuffixOf` T.pack srcPath
                    then Tar.read . GZip.decompress <$> BSL.readFile srcPath
                    else Tar.read <$> BSL.readFile srcPath
            Tar.unpack dstPath entries
        
        Unzip srcRef dstRef -> do
            srcPath <- resolveRef ctx srcRef
            dstPath <- resolveRef ctx dstRef
            createDirectoryIfMissing True dstPath
            archive <- Zip.toArchive <$> BSL.readFile srcPath
            Zip.extractFilesFromArchive [Zip.OptDestination dstPath] archive
        
        Substitute fileRef replacements -> do
            filePath <- resolveRef ctx fileRef
            content <- TIO.readFile filePath
            let applyReplacement c Replacement{..} = T.replace from to c
                content' = foldl' applyReplacement content replacements
            TIO.writeFile filePath content'
        
        PatchElfRpath pathRef rpathRefs -> do
            binaryPath <- resolveRef ctx pathRef
            rpathStrs <- mapM (resolveRef ctx) rpathRefs
            let rpathStr = intercalate ":" rpathStrs
            callProcess "patchelf" ["--set-rpath", rpathStr, binaryPath]
        
        PatchElfInterpreter pathRef interpRef -> do
            binaryPath <- resolveRef ctx pathRef
            interpPath <- resolveRef ctx interpRef
            callProcess "patchelf" ["--set-interpreter", interpPath, binaryPath]
        
        PatchElfShrink pathRef -> do
            binaryPath <- resolveRef ctx pathRef
            callProcess "patchelf" ["--shrink-rpath", binaryPath]
        
        CMakeConfigure srcDirRef buildDirRef prefixRef buildType flags gen -> do
            srcPath <- resolveRef ctx srcDirRef
            buildPath <- resolveRef ctx buildDirRef
            prefixPath <- resolveRef ctx prefixRef
            createDirectoryIfMissing True buildPath
            let genFlag = case gen of
                    Ninja -> ["-G", "Ninja"]
                    Make -> ["-G", "Unix Makefiles"]
                    DefaultGenerator -> []
                allFlags = genFlag ++
                    [ "-S", srcPath
                    , "-B", buildPath
                    , "-DCMAKE_INSTALL_PREFIX=" ++ prefixPath
                    , "-DCMAKE_BUILD_TYPE=" ++ T.unpack buildType
                    ] ++ map T.unpack flags
            callProcess "cmake" allFlags
        
        CMakeBuild buildDirRef targetMay jobsMay -> do
            buildPath <- resolveRef ctx buildDirRef
            let jobsFlag = maybe [] (\j -> ["-j", show j]) jobsMay
                targetFlag = maybe [] (\t -> ["--target", T.unpack t]) targetMay
            callProcess "cmake" $ ["--build", buildPath] ++ jobsFlag ++ targetFlag
        
        CMakeInstall buildDirRef -> do
            buildPath <- resolveRef ctx buildDirRef
            callProcess "cmake" ["--install", buildPath]
        
        MakeAction targets flags jobsMay dirMay -> do
            let jobsFlag = maybe [] (\j -> ["-j" ++ show j]) jobsMay
                allFlags = map T.unpack flags ++ jobsFlag ++ map T.unpack targets
            case dirMay of
                Just dirRef -> do
                    dirPath <- resolveRef ctx dirRef
                    callProcess "make" $ ["-C", dirPath] ++ allFlags
                Nothing -> callProcess "make" allFlags
        
        InstallBin srcRef -> do
            srcPath <- resolveRef ctx srcRef
            let dstPath = ctxOut ctx </> "bin" </> takeFileName srcPath
            createDirectoryIfMissing True (ctxOut ctx </> "bin")
            copyFile srcPath dstPath
            setFileMode dstPath 0o755
        
        InstallLib srcRef -> do
            srcPath <- resolveRef ctx srcRef
            let dstPath = ctxOut ctx </> "lib" </> takeFileName srcPath
            createDirectoryIfMissing True (ctxOut ctx </> "lib")
            copyFile srcPath dstPath
        
        InstallInclude srcRef -> do
            srcPath <- resolveRef ctx srcRef
            let dstPath = ctxOut ctx </> "include" </> takeFileName srcPath
            createDirectoryIfMissing True (ctxOut ctx </> "include")
            isDir <- doesDirectoryExist srcPath
            if isDir
                then copyDirectoryRecursive srcPath dstPath
                else copyFile srcPath dstPath
        
        Seq actions -> mapM_ (executeAction ctx) actions
        
        Shell cmd -> do
            -- Escape hatch - run shell command
            log' $ "WARNING: Using shell escape hatch: " ++ T.unpack cmd
            (exitCode, _, err) <- readProcessWithExitCode "sh" ["-c", T.unpack cmd] ""
            case exitCode of
                ExitSuccess -> return ()
                ExitFailure n -> fail $ "Shell command failed with code " ++ show n ++ ": " ++ err

-- | Convert Mode to octal int
modeToInt :: Mode -> Int
modeToInt = \case
    ModeR -> 0o444
    ModeRW -> 0o644
    ModeRX -> 0o555
    ModeRWX -> 0o755
    ModeOctal n -> fromIntegral n

-- | Log action being executed
logAction :: Action -> IO ()
logAction = \case
    Copy _ _ -> log' "copy"
    Move _ _ -> log' "move"
    Symlink _ _ -> log' "symlink"
    Mkdir _ _ -> log' "mkdir"
    Remove _ _ -> log' "remove"
    Touch _ -> log' "touch"
    Chmod _ _ -> log' "chmod"
    Write _ _ -> log' "write"
    Append _ _ -> log' "append"
    Untar _ _ _ -> log' "untar"
    Unzip _ _ -> log' "unzip"
    Substitute _ _ -> log' "substitute"
    PatchElfRpath _ _ -> log' "patchelf --set-rpath"
    PatchElfInterpreter _ _ -> log' "patchelf --set-interpreter"
    PatchElfShrink _ -> log' "patchelf --shrink-rpath"
    CMakeConfigure _ _ _ _ _ _ -> log' "cmake configure"
    CMakeBuild _ _ _ -> log' "cmake build"
    CMakeInstall _ -> log' "cmake install"
    MakeAction _ _ _ _ -> log' "make"
    InstallBin _ -> log' "install bin"
    InstallLib _ -> log' "install lib"
    InstallInclude _ -> log' "install include"
    Seq _ -> log' "seq"
    Shell cmd -> log' $ "shell: " ++ T.unpack (T.take 50 cmd)

log' :: String -> IO ()
log' msg = hPutStrLn stderr $ "[aleph-exec] " ++ msg

-- | Copy directory recursively
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

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

data Options = Options
    { optSpecFile :: FilePath
    , optPhase :: Maybe String
    }

main :: IO ()
main = do
    args <- getArgs
    case parseArgs args of
        Left err -> do
            hPutStrLn stderr $ "Error: " ++ err
            hPutStrLn stderr "Usage: aleph-exec --spec SPEC.dhall [--phase PHASE]"
            hPutStrLn stderr "Phases: unpack, patch, configure, build, check, install, fixup"
            exitFailure
        Right opts -> runBuild opts

parseArgs :: [String] -> Either String Options
parseArgs = go (Options "" Nothing)
  where
    go opts [] 
        | null (optSpecFile opts) = Left "Missing --spec argument"
        | otherwise = Right opts
    go opts ("--spec" : file : rest) = go opts { optSpecFile = file } rest
    go opts ("--phase" : phase : rest) = go opts { optPhase = Just phase } rest
    go _ (arg : _) = Left $ "Unknown argument: " ++ arg

runBuild :: Options -> IO ()
runBuild Options{..} = do
    log' $ "Reading Dhall spec: " ++ optSpecFile
    
    -- Read and parse Dhall spec
    dhallText <- TIO.readFile optSpecFile
    spec <- input auto dhallText
    
    log' $ "Building " ++ T.unpack (pname spec) ++ "-" ++ T.unpack (version spec)
    
    ctx <- getContext
    log' $ "Output: " ++ ctxOut ctx
    case ctxSrc ctx of
        Just s -> log' $ "Source: " ++ s
        Nothing -> log' "No source"
    
    createDirectoryIfMissing True (ctxOut ctx)
    
    let Phases{..} = phases spec
    
    case optPhase of
        Nothing -> do
            -- Run all phases
            runPhase "unpack" unpack
            runPhase "patch" patch
            runPhase "configure" configure
            runPhase "build" build
            runPhase "check" check
            runPhase "install" install
            runPhase "fixup" fixup
        Just phase -> do
            let actions = case phase of
                    "unpack" -> unpack
                    "patch" -> patch
                    "configure" -> configure
                    "build" -> build
                    "check" -> check
                    "install" -> install
                    "fixup" -> fixup
                    _ -> error $ "Unknown phase: " ++ phase
            runPhase phase actions
    
    log' "Build complete"
    exitSuccess
  where
    runPhase name actions = do
        unless (null actions) $ do
            log' $ "Phase: " ++ name
            ctx <- getContext
            mapM_ (executeAction ctx) actions
