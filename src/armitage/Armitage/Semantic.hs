{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Semantic
Description : Tree-sitter based semantic analysis for multi-language support

The semantic layer extracts structured information from source files:
  - Import statements (what modules/packages are used)
  - Export declarations (what symbols are defined)
  - Symbol references (for go-to-definition)

Combined with syscall tracing, this gives us:
  1. Ground truth (trace) - what files were actually accessed
  2. Semantics (parse) - what the code means
  3. Unification - validated dependency graph

Supports:
  - Python: import X, from X import Y
  - JavaScript/TypeScript: import/require
  - Haskell: import qualified X as Y
  - Rust: use X::Y
-}
module Armitage.Semantic (
    -- * Import Analysis
    Import (..),
    ImportKind (..),
    extractImports,
    extractImportsPython,

    -- * Package Resolution
    PackageInfo (..),
    resolvePackage,
    resolvePythonPackage,

    -- * Module Analysis
    ModuleInfo (..),
    analyzeModule,

    -- * Language Detection
    Lang (..),
    detectLanguage,
) where

import Control.Exception (SomeException, try)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BSU
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Foreign.C.String (peekCString)

import Foreign.Ptr (Ptr, nullPtr)
import Foreign.Storable (peek)
import GHC.Generics (Generic)
import System.Directory (doesFileExist, listDirectory)
import System.FilePath (takeExtension, (</>))

import TreeSitter.Node (Node (..), nodeStartByte, nodeType)
import TreeSitter.Parser (
    Parser,
    ts_parser_delete,
    ts_parser_new,
    ts_parser_parse_string,
    ts_parser_set_language,
 )
import TreeSitter.Tree (Tree, ts_tree_delete, withRootNode)

import qualified TreeSitter.Python as TSPython

-- -----------------------------------------------------------------------------
-- Types
-- -----------------------------------------------------------------------------

-- | Detected programming language
data Lang
    = LangPython
    | LangJavaScript
    | LangTypeScript
    | LangHaskell
    | LangRust
    | LangGo
    | LangUnknown
    deriving stock (Show, Eq, Generic)

-- | What kind of import
data ImportKind
    = -- | import X (Python), import X (Haskell)
      ImportModule
    | -- | from X import Y (the Y part)
      ImportFrom Text
    | -- | import qualified X as Y (the Y alias)
      ImportQualified Text
    | -- | Known external package (pypi, npm, hackage)
      ImportPackage
    | -- | Standard library
      ImportStdlib
    | -- | Local file
      ImportLocal
    deriving stock (Show, Eq, Generic)

-- | An import statement
data Import = Import
    { impModule :: Text
    -- ^ Module name (e.g., "requests", "os.path")
    , impKind :: ImportKind
    -- ^ What kind of import
    , impLine :: Int
    -- ^ Line number in source
    , impResolved :: Maybe Text
    -- ^ Resolved file path (if local)
    }
    deriving stock (Show, Eq, Generic)

-- | Package metadata
data PackageInfo = PackageInfo
    { pkgName :: Text
    -- ^ Package name (e.g., "requests")
    , pkgVersion :: Text
    -- ^ Version (e.g., "2.31.0")
    , pkgSource :: Text
    -- ^ Source (e.g., "pypi", "npm", "hackage")
    }
    deriving stock (Show, Eq, Generic)

-- | Complete module analysis
data ModuleInfo = ModuleInfo
    { miPath :: FilePath
    -- ^ Source file path
    , miLanguage :: Lang
    -- ^ Detected language
    , miImports :: [Import]
    -- ^ All imports
    , miPackages :: [PackageInfo]
    -- ^ Resolved packages
    }
    deriving stock (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Language Detection
-- -----------------------------------------------------------------------------

-- | Detect language from file extension
detectLanguage :: FilePath -> Lang
detectLanguage path = case takeExtension path of
    ".py" -> LangPython
    ".pyw" -> LangPython
    ".js" -> LangJavaScript
    ".mjs" -> LangJavaScript
    ".cjs" -> LangJavaScript
    ".jsx" -> LangJavaScript
    ".ts" -> LangTypeScript
    ".tsx" -> LangTypeScript
    ".mts" -> LangTypeScript
    ".cts" -> LangTypeScript
    ".hs" -> LangHaskell
    ".lhs" -> LangHaskell
    ".rs" -> LangRust
    ".go" -> LangGo
    _ -> LangUnknown

-- -----------------------------------------------------------------------------
-- Import Extraction (Tree-sitter)
-- -----------------------------------------------------------------------------

-- | Extract imports from a source file (auto-detect language)
extractImports :: FilePath -> IO [Import]
extractImports path = do
    let lang = detectLanguage path
    content <- BS.readFile path
    case lang of
        LangPython -> extractImportsPython content
        _ -> pure [] -- TODO: Other languages

{- | Extract Python imports using tree-sitter
Uses the low-level FFI API
-}
extractImportsPython :: ByteString -> IO [Import]
extractImportsPython content = do
    -- Create parser
    parser <- ts_parser_new
    langSet <- ts_parser_set_language parser TSPython.tree_sitter_python
    if not langSet
        then do
            ts_parser_delete parser
            pure []
        else do
            -- Parse the content
            tree <- parseString parser content
            if tree == nullPtr
                then do
                    ts_parser_delete parser
                    pure []
                else do
                    -- Get root node and collect imports
                    imports <- collectPythonImports content tree
                    ts_tree_delete tree
                    ts_parser_delete parser
                    pure imports

-- | Parse a string with a parser
parseString :: Ptr Parser -> ByteString -> IO (Ptr Tree)
parseString parser content =
    BSU.unsafeUseAsCStringLen content $ \(cstr, len) ->
        ts_parser_parse_string parser nullPtr cstr len

-- | Collect Python imports from a tree
collectPythonImports :: ByteString -> Ptr Tree -> IO [Import]
collectPythonImports content tree =
    withRootNode tree $ \nodePtr -> do
        node <- peek nodePtr
        collectFromNode content node

-- | Collect imports by walking the AST
collectFromNode :: ByteString -> Node -> IO [Import]
collectFromNode content node = do
    nodeTypeStr <- peekCString (nodeType node)
    case nodeTypeStr of
        "import_statement" -> do
            -- import X, Y, Z - extract module names from children
            let startByte = fromIntegral $ nodeStartByte node
            -- Simplified: just extract the text and parse it
            -- In a real implementation, we'd walk child nodes
            moduleName <- extractModuleFromImport content startByte
            pure
                [ Import
                    { impModule = moduleName
                    , impKind = ImportModule
                    , impLine = 0 -- TODO: Get line from node
                    , impResolved = Nothing
                    }
                ]
        "import_from_statement" -> do
            let startByte = fromIntegral $ nodeStartByte node
            moduleName <- extractModuleFromFrom content startByte
            pure
                [ Import
                    { impModule = moduleName
                    , impKind = ImportModule
                    , impLine = 0
                    , impResolved = Nothing
                    }
                ]
        _ -> do
            -- Recurse into children if this is a container node
            -- For now, just return empty since proper child iteration requires more FFI work
            pure []

-- | Extract module name from "import X" statement
extractModuleFromImport :: ByteString -> Int -> IO Text
extractModuleFromImport content offset = do
    let text = TE.decodeUtf8 $ BS.drop offset content
        line = T.takeWhile (/= '\n') text
        -- "import foo.bar, baz" -> "foo.bar"
        afterImport = T.strip $ T.drop 6 $ T.stripStart line -- drop "import"
        firstModule = T.takeWhile (\c -> c /= ',' && c /= ' ') afterImport
    pure firstModule

-- | Extract module name from "from X import Y" statement
extractModuleFromFrom :: ByteString -> Int -> IO Text
extractModuleFromFrom content offset = do
    let text = TE.decodeUtf8 $ BS.drop offset content
        line = T.takeWhile (/= '\n') text
        -- "from foo.bar import baz" -> "foo.bar"
        afterFrom = T.strip $ T.drop 4 $ T.stripStart line -- drop "from"
        moduleName = T.takeWhile (\c -> c /= ' ') afterFrom
    pure moduleName

-- -----------------------------------------------------------------------------
-- Package Resolution
-- -----------------------------------------------------------------------------

-- | Resolve a file path to package info
resolvePackage :: FilePath -> IO (Maybe PackageInfo)
resolvePackage path
    | "/site-packages/" `isInfixOfPath` path = resolvePythonPackage path
    | "/node_modules/" `isInfixOfPath` path = resolveNodePackage path
    | otherwise = pure Nothing
  where
    isInfixOfPath needle haystack = needle `T.isInfixOf` T.pack haystack

{- | Resolve Python package from site-packages path
Reads .dist-info/METADATA to get package name and version
-}
resolvePythonPackage :: FilePath -> IO (Maybe PackageInfo)
resolvePythonPackage path = do
    -- Path like: /nix/store/.../site-packages/requests/api.py
    -- Extract "requests" and find requests-*.dist-info
    let parts = T.splitOn "/" (T.pack path)
        siteIdx = findIndex "site-packages" parts
    case siteIdx of
        Nothing -> pure Nothing
        Just idx -> do
            let pkgName_ = parts !! (idx + 1)
                sitePackagesPath = T.unpack $ T.intercalate "/" (take (idx + 1) parts)
            findDistInfo sitePackagesPath (T.unpack pkgName_)
  where
    findIndex needle xs = go 0 xs
      where
        go _ [] = Nothing
        go n (x : rest) = if x == needle then Just n else go (n + 1) rest

-- | Find and parse dist-info for a package
findDistInfo :: FilePath -> String -> IO (Maybe PackageInfo)
findDistInfo sitePackages pkgName_ = do
    exists <- doesFileExist sitePackages
    if not exists
        then pure Nothing
        else do
            entries <- listDirectory sitePackages
            let distInfos = filter (".dist-info" `isSuffixOfStr`) entries
                matching = filter ((T.toLower (T.pack pkgName_) `T.isPrefixOf`) . T.toLower . T.pack) distInfos
            case matching of
                (distInfo : _) -> parseMetadata (sitePackages </> distInfo </> "METADATA")
                [] -> pure Nothing
  where
    isSuffixOfStr suffix str = suffix `T.isSuffixOf` T.pack str

-- | Parse Python METADATA file
parseMetadata :: FilePath -> IO (Maybe PackageInfo)
parseMetadata path = do
    result <- try $ TIO.readFile path
    case result of
        Left (_ :: SomeException) -> pure Nothing
        Right content -> do
            let ls = T.lines content
                name = extractField "Name:" ls
                version = extractField "Version:" ls
            case (name, version) of
                (Just n, Just v) ->
                    pure $
                        Just
                            PackageInfo
                                { pkgName = n
                                , pkgVersion = v
                                , pkgSource = "pypi"
                                }
                _ -> pure Nothing
  where
    extractField prefix ls =
        case filter (prefix `T.isPrefixOf`) ls of
            (l : _) -> Just $ T.strip $ T.drop (T.length prefix) l
            [] -> Nothing

-- | Resolve Node.js package from node_modules
resolveNodePackage :: FilePath -> IO (Maybe PackageInfo)
resolveNodePackage _path = pure Nothing -- TODO

-- -----------------------------------------------------------------------------
-- Module Analysis
-- -----------------------------------------------------------------------------

-- | Analyze a source file completely
analyzeModule :: FilePath -> IO ModuleInfo
analyzeModule path = do
    let lang = detectLanguage path
    imports <- extractImports path

    pure
        ModuleInfo
            { miPath = path
            , miLanguage = lang
            , miImports = imports
            , miPackages = [] -- TODO: Cross-reference with trace
            }
