{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Main
Description : Combine Abseil's ~130 static libraries into one

This script replaces combine-archive.sh with typed Haskell.
It parses pkg-config files, builds a dependency graph,
topologically sorts using Kahn's algorithm, and combines
archives using ar.

Usage: combine-archive <output-dir> [ar-prefix]
-}
module Main where

import Aleph.Script hiding (filter, head, length, lines, tail, unlines)
import qualified Aleph.Script as W
import Data.List (filter, sort)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as T
import System.Environment (getArgs)
import Prelude hiding (FilePath)

-- | Dependency graph: library name -> set of dependencies
type DepsGraph = Map.Map Text (Set.Set Text)

-- | In-degree map for Kahn's algorithm
type InDegree = Map.Map Text Int

main :: IO ()
main = script $ do
    args <- liftIO getArgs
    case args of
        [outDir] -> combineArchive outDir ""
        [outDir, arPrefix] -> combineArchive outDir arPrefix
        _ -> die "Usage: combine-archive <output-dir> [ar-prefix]"

combineArchive :: FilePath -> String -> Sh ()
combineArchive outDir arPrefix = do
    let libDir = outDir </> "lib"
        pkgconfigDir = libDir </> "pkgconfig"
        ar = arPrefix <> "ar"

    cd libDir

    echo "// extracting dependencies from pkg-config files..."

    -- Find all absl_*.pc files
    pcFiles <- findPkgConfigFiles pkgconfigDir

    -- Parse dependencies and build graph
    (graph, allLibs, privateDeps) <- buildDepsGraph pcFiles

    echo $ "// found " <> pack (show (Set.size allLibs)) <> " libraries"

    -- Topologically sort libraries
    sortedLibs <- kahnSort graph allLibs

    -- Verify we got all libraries
    actualLibs <- findAbseilArchives
    let sortedCount = length sortedLibs
        actualCount = length actualLibs

    libs <-
        if sortedCount /= actualCount
            then do
                echoErr $
                    "// warning: dependency sort found "
                        <> pack (show sortedCount)
                        <> " libraries but "
                        <> pack (show actualCount)
                        <> " exist"
                echoErr "// falling back to simple sort"
                pure $ sort actualLibs
            else pure sortedLibs

    -- Write manifest for debugging
    writeManifest libs

    -- Create ar script and combine archives
    createCombinedArchive ar libs

    -- Verify output
    exists <- test_f "libabseil.a"
    unless exists $ die "libabseil.a was not created"

    -- Cleanup individual archives
    forM_ actualLibs rm

    -- Remove individual pkg-config files
    forM_ pcFiles $ \pc -> do
        let name = basename pc
        when ("absl_" `isPrefixOf` pack name) $ rm pc

    -- Generate combined pkg-config file
    generatePkgConfig outDir privateDeps

    -- Report success
    size <- getFileSize "libabseil.a"
    echo $ "// created libabseil.a (" <> formatSize size <> ")"
    echo $ "// dependencies: " <> T.unwords (Set.toList privateDeps)
    echo "// success: archive ready for hypermodern computing"

-- | Find all absl_*.pc files in pkgconfig directory
findPkgConfigFiles :: FilePath -> Sh [FilePath]
findPkgConfigFiles dir = do
    exists <- test_d dir
    if exists
        then do
            files <- ls dir
            pure $
                filter
                    ( \f ->
                        "absl_" `isPrefixOf` pack (takeFileName f)
                            && ".pc" `isSuffixOf` pack f
                    )
                    files
        else pure []

-- | Find all libabsl_*.a files in current directory
findAbseilArchives :: Sh [FilePath]
findAbseilArchives = do
    files <- ls "."
    -- Use takeFileName to handle paths like "./libabsl_foo.a"
    pure $
        filter
            ( \f ->
                "libabsl_" `isPrefixOf` pack (takeFileName f)
                    && ".a" `isSuffixOf` pack f
            )
            files

-- | Parse pkg-config files and build dependency graph
buildDepsGraph :: [FilePath] -> Sh (DepsGraph, Set.Set Text, Set.Set Text)
buildDepsGraph pcFiles = do
    -- Collect private dependencies (pthread, m, rt, dl, etc.)
    privateDeps <- collectPrivateDeps pcFiles

    -- Build dependency graph
    results <- forM pcFiles $ \pc -> do
        let libName = pack $ dropExtension $ takeFileName pc
            libFile = "lib" <> unpack libName <> ".a"

        exists <- test_f libFile
        if exists
            then do
                deps <- extractAbslDeps pc
                pure $ Just (libName, deps)
            else pure Nothing

    let validResults = catMaybes results
        graph = Map.fromList validResults
        allLibs = Set.fromList $ map fst validResults

    pure (graph, allLibs, privateDeps)

-- | Extract Libs.private dependencies from all pkg-config files
collectPrivateDeps :: [FilePath] -> Sh (Set.Set Text)
collectPrivateDeps pcFiles = do
    deps <- forM pcFiles $ \pc -> do
        content <- liftIO $ Prelude.readFile pc
        pure $ extractPrivateLibs (pack content)

    -- Always include common system libraries
    let baseDeps = Set.fromList ["pthread", "m", "rt", "dl"]
    pure $ Set.union baseDeps (Set.unions deps)

-- | Extract -l flags from Libs.private line
extractPrivateLibs :: Text -> Set.Set Text
extractPrivateLibs content =
    let ls = T.lines content
        privLines = filter ("Libs.private:" `isInfixOf`) ls
        tokens = concatMap T.words privLines
        libs = mapMaybe extractLib tokens
     in Set.fromList libs
  where
    extractLib t
        | "-l" `isPrefixOf` t = Just (T.drop 2 t)
        | otherwise = Nothing

-- | Extract absl_* dependencies from Requires fields
extractAbslDeps :: FilePath -> Sh (Set.Set Text)
extractAbslDeps pc = do
    content <- liftIO $ Prelude.readFile pc
    let ls = T.lines (pack content)
        reqLines =
            filter
                ( \l ->
                    "Requires:" `isPrefixOf` l
                        || "Requires.private:" `isPrefixOf` l
                )
                ls
        tokens = concatMap (T.words . snd . breakOn ":") reqLines
        -- Remove commas and filter for absl_* names
        cleaned = map (W.replace "," "") tokens
        abslDeps = filter ("absl_" `isPrefixOf`) cleaned
    pure $ Set.fromList abslDeps

-- | Kahn's algorithm for topological sort
kahnSort :: DepsGraph -> Set.Set Text -> Sh [FilePath]
kahnSort graph allLibs = do
    let
        -- Initialize in-degrees to 0
        initDegrees = Map.fromList [(lib, 0) | lib <- Set.toList allLibs]

        -- Calculate in-degrees: for each lib, increment degree of its deps
        inDegrees = Map.foldrWithKey countDeps initDegrees graph

        countDeps _lib deps acc =
            Set.foldr
                ( \dep m ->
                    if Set.member dep allLibs
                        then Map.adjust (+ 1) dep m
                        else m
                )
                acc
                deps

        -- Initial queue: nodes with in-degree 0
        initialQueue = [lib | (lib, deg) <- Map.toList inDegrees, deg == 0]

    go inDegrees (sort initialQueue) []
  where
    go _ [] sorted = pure $ reverse $ map toArchive sorted
    go degrees queue sorted = do
        let current = head queue
            rest = tail queue
            sorted' = current : sorted

            -- Find all nodes that depend on current
            -- and decrement their in-degree
            (degrees', newReady) =
                Map.foldrWithKey
                    ( \lib deps (d, ready) ->
                        if Set.member current deps
                            then
                                let newDeg = Map.findWithDefault 0 lib d - 1
                                    d' = Map.insert lib newDeg d
                                    ready' =
                                        if newDeg == 0
                                            then lib : ready
                                            else ready
                                 in (d', ready')
                            else (d, ready)
                    )
                    (degrees, [])
                    graph

            -- Add newly ready nodes to queue (sorted for determinism)
            queue' = sort (rest ++ newReady)

        go degrees' queue' sorted'

    toArchive libName = "lib" <> unpack libName <> ".a"

-- | Write manifest file for debugging
writeManifest :: [FilePath] -> Sh ()
writeManifest libs = do
    cwd <- pwd
    let content = T.unlines $ ["Libraries combined:"] ++ map pack libs
    liftIO $ Prelude.writeFile (cwd </> "libabseil.manifest") (unpack content)

-- | Create combined archive using ar -M script
createCombinedArchive :: String -> [FilePath] -> Sh ()
createCombinedArchive ar libs = do
    -- Get current directory (we need absolute paths for ar)
    cwd <- pwd

    -- Write ar script with absolute paths
    let arScript =
            T.unlines $
                ["CREATE " <> pack cwd <> "/libabseil.a"]
                    ++ ["ADDLIB " <> pack cwd <> "/" <> pack lib | lib <- libs]
                    ++ ["SAVE", "END"]

    -- Write to /tmp since nix store paths may be read-only during fixup
    let arScriptPath = "/tmp/combine-archive.ar"
    liftIO $ Prelude.writeFile arScriptPath (unpack arScript)

    -- Run ar -M with stdin from file
    _ <- errExit False $ bash $ pack ar <> " -M < " <> pack arScriptPath
    code <- exitCode

    when (code /= 0) $ die "failed to combine archives"

-- | Generate combined pkg-config file
generatePkgConfig :: FilePath -> Set.Set Text -> Sh ()
generatePkgConfig outDir privateDeps = do
    let privateLibs = T.unwords ["-l" <> dep | dep <- sort $ Set.toList privateDeps]
        content =
            T.unlines
                [ "prefix=" <> pack outDir
                , "exec_prefix=${prefix}"
                , "libdir=${prefix}/lib"
                , "includedir=${prefix}/include"
                , ""
                , "Name: libabseil"
                , "Description: Abseil C++ libraries (libmodern combined archive)"
                , "Version: 20250127.1"
                , "URL: https://abseil.io/"
                , "Libs: -L${libdir} -labseil"
                , "Libs.private: -L${libdir} " <> privateLibs
                , "Cflags: -I${includedir}"
                ]

    -- Ensure pkgconfig directory exists (we may have deleted all files in it)
    -- Use absolute path since Prelude.writeFile doesn't respect Shelly's cd
    cwd <- pwd
    mkdirP "pkgconfig"
    liftIO $ Prelude.writeFile (cwd </> "pkgconfig" </> "abseil.pc") (unpack content)

-- | Get file size
getFileSize :: FilePath -> Sh Integer
getFileSize path = do
    output <- run "stat" ["-c", "%s", pack path]
    pure $ read $ unpack $ strip output

-- | Format file size for display
formatSize :: Integer -> Text
formatSize bytes
    | bytes >= 1024 * 1024 = pack (show (bytes `div` (1024 * 1024))) <> "MB"
    | bytes >= 1024 = pack (show (bytes `div` 1024)) <> "KB"
    | otherwise = pack (show bytes) <> "B"
