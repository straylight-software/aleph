{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Shim
Description : Read metadata from shim-generated ELF files

The shim compilers (cc-shim, ld-shim, ar-shim) emit fake ELF files
with build metadata encoded as symbols in the .data section.

This module reads that metadata to reconstruct:
  - What source files were compiled
  - What include paths were used
  - What libraries were linked
  - The complete dependency graph

This is the key insight: run any build system with shims,
get perfect dependency information instantly.
-}
module Armitage.Shim (
    -- * Metadata types
    CompileInfo (..),
    LinkInfo (..),
    ArchiveInfo (..),
    BuildMetadata (..),

    -- * Reading metadata
    readObjectMetadata,
    readExecutableMetadata,
    readArchiveMetadata,
    readBuildMetadata,

    -- * Log parsing
    ShimLogEntry (..),
    parseShimLog,

    -- * Shim generation
    generateShimEnv,
    ShimPaths (..),

    -- * Full analysis pipeline
    AnalysisResult (..),
    AnalysisConfig (..),
    defaultAnalysisConfig,
    shimAnalyze,
    ValidationResult (..),
) where

import Control.Exception (SomeException, try)
import Control.Monad (filterM, forM, forM_, when)
import Data.Bits (shiftL, (.|.))
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Data.Word (Word16, Word32, Word64)
import System.Directory (createDirectoryIfMissing, doesDirectoryExist, doesFileExist, listDirectory)
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import System.Process (CreateProcess (..), proc, readCreateProcessWithExitCode, readProcessWithExitCode)

-- -----------------------------------------------------------------------------
-- Metadata Types
-- -----------------------------------------------------------------------------

-- | Metadata from a compiled object file
data CompileInfo = CompileInfo
    { ciSources :: [Text]
    -- ^ Source files compiled
    , ciIncludes :: [Text]
    -- ^ Include paths (-I, -isystem)
    , ciDefines :: [Text]
    -- ^ Preprocessor defines (-D)
    , ciFlags :: [Text]
    -- ^ Other compiler flags
    , ciOutput :: Text
    -- ^ Output file
    }
    deriving (Show, Eq)

-- | Metadata from a linked executable/library
data LinkInfo = LinkInfo
    { liObjects :: [Text]
    -- ^ Object files linked
    , liLibs :: [Text]
    -- ^ Libraries linked (-l)
    , liLibPaths :: [Text]
    -- ^ Library search paths (-L)
    , liRPaths :: [Text]
    -- ^ Runtime paths (-rpath)
    , liOutput :: Text
    {- ^ Output file
    Aggregated from all compiled objects:
    -}
    , liAllSources :: [Text]
    , liAllIncludes :: [Text]
    , liAllDefines :: [Text]
    , liAllFlags :: [Text]
    }
    deriving (Show, Eq)

-- | Metadata from an archive
data ArchiveInfo = ArchiveInfo
    { aiArchive :: Text
    -- ^ Archive name
    , aiMembers :: [Text]
    -- ^ Member objects
    , aiAllSources :: [Text]
    , aiAllIncludes :: [Text]
    , aiAllDefines :: [Text]
    , aiAllFlags :: [Text]
    }
    deriving (Show, Eq)

-- | Complete build metadata for a target
data BuildMetadata = BuildMetadata
    { bmTarget :: Text
    -- ^ Final output
    , bmSources :: [Text]
    -- ^ All source files
    , bmIncludes :: [Text]
    -- ^ All include paths
    , bmDefines :: [Text]
    -- ^ All defines
    , bmFlags :: [Text]
    -- ^ All flags
    , bmLibs :: [Text]
    -- ^ All libraries
    , bmLibPaths :: [Text]
    -- ^ All library paths
    , bmObjects :: [Text]
    -- ^ All intermediate objects
    }
    deriving (Show, Eq)

-- -----------------------------------------------------------------------------
-- ELF Parsing (minimal, just enough to read symbols)
-- -----------------------------------------------------------------------------

-- | Read a little-endian Word16
readWord16 :: ByteString -> Int -> Word16
readWord16 bs off =
    fromIntegral (BS.index bs off)
        .|. (fromIntegral (BS.index bs (off + 1)) `shiftL` 8)

-- | Read a little-endian Word32
readWord32 :: ByteString -> Int -> Word32
readWord32 bs off =
    fromIntegral (BS.index bs off)
        .|. (fromIntegral (BS.index bs (off + 1)) `shiftL` 8)
        .|. (fromIntegral (BS.index bs (off + 2)) `shiftL` 16)
        .|. (fromIntegral (BS.index bs (off + 3)) `shiftL` 24)

-- | Read a little-endian Word64
readWord64 :: ByteString -> Int -> Word64
readWord64 bs off =
    fromIntegral (BS.index bs off)
        .|. (fromIntegral (BS.index bs (off + 1)) `shiftL` 8)
        .|. (fromIntegral (BS.index bs (off + 2)) `shiftL` 16)
        .|. (fromIntegral (BS.index bs (off + 3)) `shiftL` 24)
        .|. (fromIntegral (BS.index bs (off + 4)) `shiftL` 32)
        .|. (fromIntegral (BS.index bs (off + 5)) `shiftL` 40)
        .|. (fromIntegral (BS.index bs (off + 6)) `shiftL` 48)
        .|. (fromIntegral (BS.index bs (off + 7)) `shiftL` 56)

-- | Read null-terminated string from ByteString
readCString :: ByteString -> Int -> Text
readCString bs off =
    let bytes = BS.takeWhile (/= 0) (BS.drop off bs)
     in TE.decodeUtf8 bytes

-- | Extract armitage symbols from an ELF file
readElfSymbols :: ByteString -> Map Text Text
readElfSymbols bs
    | BS.length bs < 64 = Map.empty -- Too small for ELF header
    | BS.take 4 bs /= "\x7fELF" = Map.empty -- Not ELF
    | otherwise =
        let shoff = fromIntegral $ readWord64 bs 40 -- e_shoff
            shentsize = fromIntegral $ readWord16 bs 58 -- e_shentsize
            shnum = fromIntegral $ readWord16 bs 60 -- e_shnum

            -- Find .symtab section
            findSymtab i
                | i >= shnum = Nothing
                | otherwise =
                    let shdrOff = shoff + i * shentsize
                        shType = readWord32 bs (shdrOff + 4) -- sh_type at offset 4
                     in if shType == 2 -- SHT_SYMTAB
                            then Just (shdrOff, i)
                            else findSymtab (i + 1)
         in case findSymtab 0 of
                Nothing -> Map.empty
                Just (symtabShdr, _) ->
                    let symtabOff = fromIntegral $ readWord64 bs (symtabShdr + 24) -- sh_offset
                        symtabSize = fromIntegral $ readWord64 bs (symtabShdr + 32) -- sh_size
                        strtabIdx = fromIntegral $ readWord32 bs (symtabShdr + 40) -- sh_link
                        strtabShdr = shoff + strtabIdx * shentsize
                        strtabOff = fromIntegral $ readWord64 bs (strtabShdr + 24)

                        -- Find .data section for symbol values
                        findData i
                            | i >= shnum = Nothing
                            | otherwise =
                                let shdrOff = shoff + i * shentsize
                                    shType = readWord32 bs (shdrOff + 4)
                                    shFlags = readWord64 bs (shdrOff + 8)
                                 in if shType == 1 && shFlags == 3 -- SHT_PROGBITS, SHF_WRITE|SHF_ALLOC
                                        then Just (fromIntegral $ readWord64 bs (shdrOff + 24))
                                        else findData (i + 1)

                        dataOff = fromMaybe 0 (findData 0)

                        -- Parse symbols
                        numSyms = symtabSize `div` 24 -- sizeof(Elf64_Sym)
                        parseSyms acc i
                            | i >= numSyms = acc
                            | otherwise =
                                let symOff = symtabOff + i * 24
                                    stName = fromIntegral $ readWord32 bs symOff -- st_name
                                    stValue = fromIntegral $ readWord64 bs (symOff + 8) -- st_value
                                    name = readCString bs (strtabOff + stName)
                                    value = readCString bs (dataOff + stValue)
                                 in if "__armitage_" `T.isPrefixOf` name
                                        then parseSyms (Map.insert name value acc) (i + 1)
                                        else parseSyms acc (i + 1)
                     in parseSyms Map.empty 0

-- | Split colon-separated string
splitColons :: Text -> [Text]
splitColons t = filter (not . T.null) $ T.splitOn ":" t

-- -----------------------------------------------------------------------------
-- Reading Metadata
-- -----------------------------------------------------------------------------

-- | Read compile metadata from an object file
readObjectMetadata :: FilePath -> IO (Maybe CompileInfo)
readObjectMetadata path = do
    exists <- doesFileExist path
    if not exists
        then pure Nothing
        else do
            result <- try $ BS.readFile path
            case result of
                Left (_ :: SomeException) -> pure Nothing
                Right bs ->
                    let syms = readElfSymbols bs
                     in if Map.null syms
                            then pure Nothing
                            else
                                pure $
                                    Just
                                        CompileInfo
                                            { ciSources = splitColons $ Map.findWithDefault "" "__armitage_sources" syms
                                            , ciIncludes = splitColons $ Map.findWithDefault "" "__armitage_includes" syms
                                            , ciDefines = splitColons $ Map.findWithDefault "" "__armitage_defines" syms
                                            , ciFlags = splitColons $ Map.findWithDefault "" "__armitage_flags" syms
                                            , ciOutput = Map.findWithDefault "" "__armitage_output" syms
                                            }

-- | Read link metadata from an executable/library
readExecutableMetadata :: FilePath -> IO (Maybe LinkInfo)
readExecutableMetadata path = do
    exists <- doesFileExist path
    if not exists
        then pure Nothing
        else do
            result <- try $ BS.readFile path
            case result of
                Left (_ :: SomeException) -> pure Nothing
                Right bs ->
                    let syms = readElfSymbols bs
                     in if Map.null syms
                            then pure Nothing
                            else
                                pure $
                                    Just
                                        LinkInfo
                                            { liObjects = splitColons $ Map.findWithDefault "" "__armitage_objects" syms
                                            , liLibs = splitColons $ Map.findWithDefault "" "__armitage_libs" syms
                                            , liLibPaths = splitColons $ Map.findWithDefault "" "__armitage_libpaths" syms
                                            , liRPaths = splitColons $ Map.findWithDefault "" "__armitage_rpaths" syms
                                            , liOutput = Map.findWithDefault "" "__armitage_output" syms
                                            , liAllSources = splitColons $ Map.findWithDefault "" "__armitage_all_sources" syms
                                            , liAllIncludes = splitColons $ Map.findWithDefault "" "__armitage_all_includes" syms
                                            , liAllDefines = splitColons $ Map.findWithDefault "" "__armitage_all_defines" syms
                                            , liAllFlags = splitColons $ Map.findWithDefault "" "__armitage_all_flags" syms
                                            }

-- | Read archive metadata from a .a file
readArchiveMetadata :: FilePath -> IO (Maybe ArchiveInfo)
readArchiveMetadata path = do
    exists <- doesFileExist path
    if not exists
        then pure Nothing
        else do
            result <- try $ BS.readFile path
            case result of
                Left (_ :: SomeException) -> pure Nothing
                Right bs ->
                    let syms = readElfSymbols bs
                     in if Map.null syms
                            then pure Nothing
                            else
                                pure $
                                    Just
                                        ArchiveInfo
                                            { aiArchive = Map.findWithDefault "" "__armitage_archive" syms
                                            , aiMembers = splitColons $ Map.findWithDefault "" "__armitage_members" syms
                                            , aiAllSources = splitColons $ Map.findWithDefault "" "__armitage_all_sources" syms
                                            , aiAllIncludes = splitColons $ Map.findWithDefault "" "__armitage_all_includes" syms
                                            , aiAllDefines = splitColons $ Map.findWithDefault "" "__armitage_all_defines" syms
                                            , aiAllFlags = splitColons $ Map.findWithDefault "" "__armitage_all_flags" syms
                                            }

-- | Read complete build metadata from a final output
readBuildMetadata :: FilePath -> IO (Maybe BuildMetadata)
readBuildMetadata path = do
    linkInfo <- readExecutableMetadata path
    case linkInfo of
        Nothing -> pure Nothing
        Just li ->
            pure $
                Just
                    BuildMetadata
                        { bmTarget = liOutput li
                        , bmSources = liAllSources li
                        , bmIncludes = liAllIncludes li
                        , bmDefines = liAllDefines li
                        , bmFlags = liAllFlags li
                        , bmLibs = liLibs li
                        , bmLibPaths = liLibPaths li
                        , bmObjects = liObjects li
                        }

-- -----------------------------------------------------------------------------
-- Log Parsing
-- -----------------------------------------------------------------------------

-- | A single entry from the shim log
data ShimLogEntry = ShimLogEntry
    { sleTimestamp :: Text
    , slePid :: Int
    , sleTool :: Text
    -- ^ CC, LD, AR
    , sleArgs :: [Text]
    }
    deriving (Show, Eq)

-- | Parse the shim log file
parseShimLog :: FilePath -> IO [ShimLogEntry]
parseShimLog path = do
    exists <- doesFileExist path
    if not exists
        then pure []
        else do
            content <- TIO.readFile path
            pure $ catMaybes $ map parseLogLine $ T.lines content

parseLogLine :: Text -> Maybe ShimLogEntry
parseLogLine line = do
    -- Format: [YYYY-MM-DD HH:MM:SS] pid=NNN TOOL args...
    let stripped = T.strip line
    if T.null stripped
        then Nothing
        else do
            -- Extract timestamp
            let (ts, rest) = T.breakOn "]" stripped
                timestamp = T.drop 1 ts -- drop leading [
                afterTs = T.strip $ T.drop 1 rest -- drop ]

            -- Extract pid
            let (pidPart, rest2) = T.breakOn " " afterTs
            pid <-
                if "pid=" `T.isPrefixOf` pidPart
                    then readMaybe $ T.unpack $ T.drop 4 pidPart
                    else Nothing

            let afterPid = T.strip rest2
                (tool, argsStr) = T.breakOn " " afterPid
                args = T.words argsStr

            Just
                ShimLogEntry
                    { sleTimestamp = timestamp
                    , slePid = pid
                    , sleTool = tool
                    , sleArgs = args
                    }
  where
    readMaybe :: (Read a) => String -> Maybe a
    readMaybe s = case reads s of
        [(x, "")] -> Just x
        _ -> Nothing

-- -----------------------------------------------------------------------------
-- Shim Generation
-- -----------------------------------------------------------------------------

-- | Paths to shim binaries
data ShimPaths = ShimPaths
    { spCC :: FilePath
    , spCXX :: FilePath
    , spLD :: FilePath
    , spAR :: FilePath
    , spLogPath :: FilePath
    }
    deriving (Show, Eq)

-- | Generate environment variables to use shims
generateShimEnv :: ShimPaths -> [(String, String)]
generateShimEnv ShimPaths{..} =
    [ ("CC", spCC)
    , ("CXX", spCXX)
    , ("LD", spLD)
    , ("AR", spAR)
    , ("ARMITAGE_SHIM_LOG", spLogPath)
    , -- Also override via PATH-style vars that some build systems use
      ("CMAKE_C_COMPILER", spCC)
    , ("CMAKE_CXX_COMPILER", spCXX)
    , ("CMAKE_AR", spAR)
    , ("CMAKE_LINKER", spLD)
    ]

-- -----------------------------------------------------------------------------
-- Full Analysis Pipeline
-- -----------------------------------------------------------------------------

-- | Configuration for shim analysis
data AnalysisConfig = AnalysisConfig
    { acShimDir :: FilePath
    -- ^ Directory containing shim binaries
    , acLogDir :: FilePath
    -- ^ Directory for logs
    , acValidate :: Bool
    -- ^ Run strace validation
    , acVerbose :: Bool
    -- ^ Verbose output
    , acKeepTemp :: Bool
    -- ^ Keep temporary files
    }
    deriving (Show, Eq)

defaultAnalysisConfig :: AnalysisConfig
defaultAnalysisConfig =
    AnalysisConfig
        { acShimDir = "/tmp/armitage-shims"
        , acLogDir = "/tmp/armitage"
        , acValidate = True
        , acVerbose = False
        , acKeepTemp = False
        }

-- | Result of analyzing a flake reference
data AnalysisResult = AnalysisResult
    { arFlakeRef :: Text
    -- ^ Original flake reference
    , arDrvPath :: Text
    -- ^ Resolved derivation path
    , arOutputPath :: Maybe Text
    -- ^ Build output path (if built)
    , arSources :: [Text]
    -- ^ All source files
    , arIncludes :: [Text]
    -- ^ All include paths
    , arDefines :: [Text]
    -- ^ All preprocessor defines
    , arLibs :: [Text]
    -- ^ All libraries linked
    , arLibPaths :: [Text]
    -- ^ All library search paths
    , arShimLog :: [ShimLogEntry]
    -- ^ Raw shim invocations
    , arValidation :: Maybe ValidationResult
    -- ^ Strace validation result
    , arErrors :: [Text]
    -- ^ Any errors encountered
    }
    deriving (Show, Eq)

{- | Result of validating shim output against strace

The validation contract is simple:
  If strace sees an artifact written to $out, shim MUST have caught it.
  Failure is not a "discrepancy" - it's a hard error.
-}
data ValidationResult = ValidationResult
    { vrMatches :: Bool
    -- ^ Did shim catch everything? FALSE = FAILURE
    , vrShimOutputs :: Set Text
    -- ^ What shim claims to have produced
    , vrStraceOutputs :: Set Text
    -- ^ All files strace saw written to output
    , vrStraceArtifacts :: Set Text
    -- ^ Subset that look like compiled artifacts
    , vrMissedArtifacts :: Set Text
    -- ^ FAILURES: artifacts strace saw, shim missed
    }
    deriving (Show, Eq)

{- | Main analysis entry point: analyze a flake reference

Architecture:
  Run build ONCE under strace with shims installed.
  Shims produce tagged outputs (ELF with __armitage_* symbols).
  After build, scan all binary artifacts in output.
  Any binary NOT tagged by shim = we missed a compiler = FAIL.

One build. Complete coverage. No guessing.
-}
shimAnalyze :: AnalysisConfig -> Text -> IO (Either Text AnalysisResult)
shimAnalyze cfg flakeRef = do
    when (acVerbose cfg) $
        TIO.putStrLn $
            "Analyzing: " <> flakeRef

    -- Resolve to derivation path
    drvResult <- resolveFlakeRef flakeRef
    case drvResult of
        Left err -> pure $ Left $ "Failed to resolve " <> flakeRef <> ": " <> err
        Right drvPath -> runAnalysis cfg flakeRef drvPath

-- | Run the actual analysis after resolving the flake ref
runAnalysis :: AnalysisConfig -> Text -> Text -> IO (Either Text AnalysisResult)
runAnalysis cfg flakeRef drvPath = do
    when (acVerbose cfg) $
        TIO.putStrLn $
            "Derivation: " <> drvPath

    let shimPaths =
            ShimPaths
                { spCC = acShimDir cfg </> "cc"
                , spCXX = acShimDir cfg </> "c++"
                , spLD = acShimDir cfg </> "ld"
                , spAR = acShimDir cfg </> "ar"
                , spLogPath = acLogDir cfg </> "shim.log"
                }
        straceLogPath = acLogDir cfg </> "strace.log"

    createDirectoryIfMissing True (acLogDir cfg)

    -- Check shims exist
    shimsExist <- doesFileExist (spCC shimPaths)
    if not shimsExist
        then pure $ Left $ "Shims not found at " <> T.pack (acShimDir cfg)
        else do
            -- Clear logs
            writeFile (spLogPath shimPaths) ""
            writeFile straceLogPath ""

            when (acVerbose cfg) $
                putStrLn "Running build with shims under strace..."

            -- Run build with shims UNDER strace - one build captures everything
            buildResult <- runShimBuildWithStrace cfg shimPaths straceLogPath drvPath
            case buildResult of
                Left err -> pure $ Left err
                Right outputPath -> do
                    logEntries <- parseShimLog (spLogPath shimPaths)
                    validation <- validateOutputs cfg outputPath
                    metadata <- case outputPath of
                        Just p -> aggregateShimMetadata (T.unpack p)
                        Nothing -> pure Nothing

                    pure $
                        Right
                            AnalysisResult
                                { arFlakeRef = flakeRef
                                , arDrvPath = drvPath
                                , arOutputPath = outputPath
                                , arSources = maybe [] bmSources metadata
                                , arIncludes = maybe [] bmIncludes metadata
                                , arDefines = maybe [] bmDefines metadata
                                , arLibs = maybe [] bmLibs metadata
                                , arLibPaths = maybe [] bmLibPaths metadata
                                , arShimLog = logEntries
                                , arValidation = Just validation
                                , arErrors = []
                                }

-- | Resolve a flake reference to its derivation path
resolveFlakeRef :: Text -> IO (Either Text Text)
resolveFlakeRef ref = do
    (code, out, err) <-
        readProcessWithExitCode
            "nix"
            ["path-info", "--derivation", T.unpack ref]
            ""
    case code of
        ExitSuccess -> pure $ Right $ T.strip $ T.pack out
        ExitFailure _ -> pure $ Left $ T.pack err

{- | Run build with shims under strace
One build, complete capture
-}
runShimBuildWithStrace ::
    AnalysisConfig ->
    ShimPaths ->
    -- | strace log path
    FilePath ->
    -- | derivation path
    Text ->
    IO (Either Text (Maybe Text))
runShimBuildWithStrace cfg shimPaths straceLogPath drvPath = do
    currentEnv <- getEnvironment
    let shimEnvVars = generateShimEnv shimPaths
        pathEnv = case lookup "PATH" currentEnv of
            Just p -> ("PATH", acShimDir cfg <> ":" <> p)
            Nothing -> ("PATH", acShimDir cfg)
        fullEnv = pathEnv : shimEnvVars ++ currentEnv

    when (acVerbose cfg) $ do
        putStrLn "Environment:"
        forM_ (take 5 shimEnvVars) $ \(k, v) ->
            putStrLn $ "  " <> k <> "=" <> v

    -- Run nix-build with shims, wrapped in strace
    -- strace captures all syscalls, shims tag their outputs
    let p =
            ( proc
                "strace"
                [ "-f"
                , "-e"
                , "trace=openat,open,creat,rename,link,symlink"
                , "-o"
                , straceLogPath
                , "--"
                , "nix-build"
                , T.unpack drvPath
                , "--no-out-link"
                ]
            )
                { env = Just fullEnv
                }

    (exitCode, stdout, _stderr) <- readCreateProcessWithExitCode p ""

    case exitCode of
        ExitSuccess -> do
            let outPath = T.strip $ T.pack stdout
            pure $ Right $ if T.null outPath then Nothing else Just outPath
        ExitFailure _ ->
            -- Build failed, but we still have partial data
            pure $ Right Nothing

{- | Validate outputs: scan all binaries, check for shim tags

Every binary artifact in output must have __armitage_* symbols.
If it doesn't, we missed a compiler. That's a FAILURE.
-}
validateOutputs :: AnalysisConfig -> Maybe Text -> IO ValidationResult
validateOutputs cfg outputPath = do
    case outputPath of
        Nothing ->
            pure
                ValidationResult
                    { vrMatches = True -- No output = nothing to validate
                    , vrShimOutputs = Set.empty
                    , vrStraceOutputs = Set.empty
                    , vrStraceArtifacts = Set.empty
                    , vrMissedArtifacts = Set.empty
                    }
        Just outPath -> do
            -- Find all binary files in output
            allFiles <- findFilesRecursive (T.unpack outPath)
            binaries <- filterByMagic (map T.pack allFiles)

            -- Check each binary for shim tags
            tagged <- filterM (hasShimTags . T.unpack) (Set.toList binaries)
            let taggedSet = Set.fromList tagged
                untagged = binaries `Set.difference` taggedSet

            when (acVerbose cfg && not (Set.null untagged)) $ do
                putStrLn "UNTAGGED BINARIES (missed by shim):"
                forM_ (Set.toList untagged) $ \f ->
                    TIO.putStrLn $ "  " <> f

            pure
                ValidationResult
                    { vrMatches = Set.null untagged
                    , vrShimOutputs = taggedSet
                    , vrStraceOutputs = binaries
                    , vrStraceArtifacts = binaries
                    , vrMissedArtifacts = untagged
                    }

-- | Check if a file has shim tags (__armitage_* symbols)
hasShimTags :: FilePath -> IO Bool
hasShimTags path = do
    result <- try $ BS.readFile path
    case result of
        Left (_ :: SomeException) -> pure False
        Right content -> pure $ hasArmitageSymbols content
  where
    hasArmitageSymbols bs = "__armitage_" `BC.isInfixOf` bs

-- | Find all files recursively in a directory
findFilesRecursive :: FilePath -> IO [FilePath]
findFilesRecursive dir = do
    exists <- doesDirectoryExist dir
    if not exists
        then pure []
        else do
            entries <- listDirectory dir
            paths <- forM entries $ \entry -> do
                let path = dir </> entry
                isDir <- doesDirectoryExist path
                if isDir
                    then findFilesRecursive path
                    else pure [path]
            pure (concat paths)

-- | Aggregate metadata from all shim-tagged files in output
aggregateShimMetadata :: FilePath -> IO (Maybe BuildMetadata)
aggregateShimMetadata outPath = do
    allFiles <- findFilesRecursive outPath
    binaries <- filterByMagic (map T.pack allFiles)

    -- Read metadata from each tagged binary
    metadatas <- forM (Set.toList binaries) $ \f -> do
        tagged <- hasShimTags (T.unpack f)
        if tagged
            then readBuildMetadata (T.unpack f)
            else pure Nothing

    -- Merge all metadata
    let validMetadata = catMaybes metadatas
    if null validMetadata
        then pure Nothing
        else pure $ Just $ mergeMetadata validMetadata

-- | Merge multiple BuildMetadata into one
mergeMetadata :: [BuildMetadata] -> BuildMetadata
mergeMetadata [] = BuildMetadata "" [] [] [] [] [] [] []
mergeMetadata (m : ms) =
    BuildMetadata
        { bmTarget = bmTarget m
        , bmSources = nub $ concatMap bmSources (m : ms)
        , bmIncludes = nub $ concatMap bmIncludes (m : ms)
        , bmDefines = nub $ concatMap bmDefines (m : ms)
        , bmFlags = nub $ concatMap bmFlags (m : ms)
        , bmLibs = nub $ concatMap bmLibs (m : ms)
        , bmLibPaths = nub $ concatMap bmLibPaths (m : ms)
        , bmObjects = nub $ concatMap bmObjects (m : ms)
        }
  where
    nub = Set.toList . Set.fromList

{- | Filter paths to only those with binary file magic
Checks actual file content, not extensions
-}
filterByMagic :: [Text] -> IO (Set Text)
filterByMagic paths = do
    results <- forM paths $ \path -> do
        isBinary <- hasBinaryMagic (T.unpack path)
        pure $ if isBinary then Just path else Nothing
    pure $ Set.fromList $ catMaybes results

{- | Check if file has binary magic bytes
Returns True for ELF, Mach-O, PE, JVM class, WASM, ar archives, etc.
-}
hasBinaryMagic :: FilePath -> IO Bool
hasBinaryMagic path = do
    exists <- doesFileExist path
    if not exists
        then pure False
        else do
            result <- try $ BS.readFile path
            case result of
                Left (_ :: SomeException) -> pure False
                Right content -> pure $ checkMagic content
  where
    checkMagic bs
        | BS.length bs < 4 = False
        | otherwise =
            let magic4 = BS.take 4 bs
                magic2 = BS.take 2 bs
             in magic4 == elfMagic -- ELF
                    || magic4 == machO32Magic -- Mach-O 32-bit
                    || magic4 == machO64Magic -- Mach-O 64-bit
                    || magic4 == machOFatMagic -- Mach-O fat/universal
                    || magic2 == peMagic -- PE (MZ)
                    || magic4 == jvmMagic -- JVM .class
                    || magic4 == wasmMagic -- WebAssembly
                    || magic4 == llvmBcMagic -- LLVM bitcode
                    || BS.take 8 bs == arMagic -- ar archive (.a)
                    || magic4 == pythonMagic2 -- Python 2 bytecode
                    || isPythonMagic bs -- Python 3 bytecode (varies)
    elfMagic = BS.pack [0x7f, 0x45, 0x4c, 0x46] -- \x7fELF
    machO32Magic = BS.pack [0xfe, 0xed, 0xfa, 0xce] -- feedface
    machO64Magic = BS.pack [0xfe, 0xed, 0xfa, 0xcf] -- feedfacf
    machOFatMagic = BS.pack [0xca, 0xfe, 0xba, 0xbe] -- cafebabe (also JVM!)
    peMagic = BS.pack [0x4d, 0x5a] -- MZ
    jvmMagic = BS.pack [0xca, 0xfe, 0xba, 0xbe] -- cafebabe
    wasmMagic = BS.pack [0x00, 0x61, 0x73, 0x6d] -- \0asm
    llvmBcMagic = BS.pack [0x42, 0x43, 0xc0, 0xde] -- BC..
    arMagic = BC.pack "!<arch>\n" -- ar archive
    pythonMagic2 = BS.pack [0x03, 0xf3, 0x0d, 0x0a] -- Python 2.7

    -- Python 3 magic varies by version, check pattern
    isPythonMagic bs =
        BS.length bs >= 4
            && BS.index bs 2 == 0x0d
            && BS.index bs 3 == 0x0a
