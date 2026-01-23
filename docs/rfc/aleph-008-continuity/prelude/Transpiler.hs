{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Dhall → Starlark/Nix Transpiler

This is the bridge from typed Dhall BUILD files to:
1. Starlark BUCK files (for Buck2)
2. Nix expressions (for Nix, via dhall-nix or direct)

The key insight: both targets are just data. No complex logic needed.
-}
module Straylight.Transpiler (
    -- * Core transpilation
    toStarlark,
    toNix,
    toBuck,

    -- * Target types
    Target (..),
    RuleKind (..),

    -- * CLI
    main,
) where

import qualified Dhall
import qualified Dhall.Core as D
import qualified Dhall.Map as DM
import qualified Dhall.Nix as DN -- From dhall-nix package

import Data.List (intercalate)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs)
import System.FilePath (takeDirectory, (-<.>), (</>))

-- =============================================================================
-- Core Types
-- =============================================================================

-- | A build target extracted from Dhall
data Target = Target
    { targetRule :: RuleKind
    , targetName :: Text
    , targetAttrs :: D.Expr D.Src D.Void
    }
    deriving (Show)

-- | Rule kinds we support
data RuleKind
    = RustLibrary
    | RustBinary
    | RustTest
    | LeanLibrary
    | LeanBinary
    | LeanTest
    | ProvenLibrary
    | CxxLibrary
    | CxxBinary
    | CxxTest
    | HaskellLibrary
    | HaskellBinary
    | HaskellTest
    | PureScriptLibrary
    | PureScriptBundle
    | PureScriptTest
    | NvLibrary
    | NvBinary
    | NvTest
    | Alias
    | Filegroup
    deriving (Show, Eq)

-- =============================================================================
-- Dhall → Starlark
-- =============================================================================

-- | Convert a Dhall expression to Starlark syntax
toStarlark :: D.Expr s a -> Text
toStarlark = \case
    -- Records become dicts or kwargs
    D.RecordLit fields ->
        let pairs =
                map
                    (\(k, v) -> k <> " = " <> toStarlark (D.recordFieldValue v))
                    (DM.toList fields)
         in "{" <> T.intercalate ", " pairs <> "}"
    -- Lists become lists
    D.ListLit _ items ->
        let elems = map toStarlark (toList items)
         in "[" <> T.intercalate ", " elems <> "]"
    -- Text becomes strings
    D.TextLit chunks ->
        "\"" <> escapeStarlark (textFromChunks chunks) <> "\""
    -- Booleans
    D.BoolLit True -> "True"
    D.BoolLit False -> "False"
    -- Numbers
    D.NaturalLit n -> T.pack (show n)
    D.IntegerLit n -> T.pack (show n)
    D.DoubleLit n -> T.pack (show (D.denote n))
    -- Optional
    D.Some inner -> toStarlark inner
    D.App D.None _ -> "None"
    -- Unions (enums) - just use the tag as a string
    D.Field (D.Union _) (D.FieldSelection _ name _) ->
        "\"" <> name <> "\""
    -- Union application (variant with value)
    D.App (D.Field (D.Union _) (D.FieldSelection _ name _)) value ->
        -- For complex variants, emit as struct
        name <> "(" <> toStarlark value <> ")"
    -- Variables shouldn't appear in normalized output
    D.Var _ -> error "Unexpected variable in normalized Dhall"
    -- Fallback
    other -> "# TODO: " <> T.pack (show other)
  where
    toList = foldr (:) []

-- | Escape a string for Starlark
escapeStarlark :: Text -> Text
escapeStarlark = T.concatMap $ \case
    '\\' -> "\\\\"
    '"' -> "\\\""
    '\n' -> "\\n"
    '\r' -> "\\r"
    '\t' -> "\\t"
    c -> T.singleton c

-- | Extract text from Dhall text literal chunks
textFromChunks :: D.Chunks s a -> Text
textFromChunks (D.Chunks [] t) = t
textFromChunks (D.Chunks ((t, _) : rest) final) =
    t <> textFromChunks (D.Chunks rest final)

-- =============================================================================
-- Dhall → Nix
-- =============================================================================

{- | Convert a Dhall expression to Nix syntax
Uses dhall-nix for the heavy lifting
-}
toNix :: D.Expr D.Src D.Void -> Text
toNix expr = DN.dhallToNix expr -- dhall-nix does the work

-- =============================================================================
-- Emit BUCK file
-- =============================================================================

-- | Emit a complete BUCK file from a list of targets
toBuck :: [Target] -> Text
toBuck targets =
    T.unlines $
        [ "# Generated from BUILD.dhall - DO NOT EDIT"
        , "# Regenerate with: straylight gen"
        , ""
        ]
            ++ concatMap emitTarget targets

-- | Emit a single target
emitTarget :: Target -> [Text]
emitTarget Target{..} =
    [ ruleName targetRule <> "("
    , "    name = \"" <> targetName <> "\","
    ]
        ++ attrLines
        ++ [")"]
  where
    attrLines = case targetAttrs of
        D.RecordLit fields ->
            [ "    " <> k <> " = " <> toStarlark (D.recordFieldValue v) <> ","
            | (k, v) <- DM.toList fields
            , k /= "common" -- Skip common, we extracted name from it
            ]
        _ -> []

-- | Rule name in Starlark
ruleName :: RuleKind -> Text
ruleName = \case
    RustLibrary -> "rust_library"
    RustBinary -> "rust_binary"
    RustTest -> "rust_test"
    LeanLibrary -> "lean_library"
    LeanBinary -> "lean_binary"
    LeanTest -> "lean_test"
    ProvenLibrary -> "proven_library"
    CxxLibrary -> "cxx_library"
    CxxBinary -> "cxx_binary"
    CxxTest -> "cxx_test"
    HaskellLibrary -> "haskell_library"
    HaskellBinary -> "haskell_binary"
    HaskellTest -> "haskell_test"
    PureScriptLibrary -> "purescript_library"
    PureScriptBundle -> "purescript_bundle"
    PureScriptTest -> "purescript_test"
    NvLibrary -> "nv_library"
    NvBinary -> "nv_binary"
    NvTest -> "nv_test"
    Alias -> "alias"
    Filegroup -> "filegroup"

-- =============================================================================
-- Parse targets from Dhall
-- =============================================================================

{- | Extract targets from a normalized Dhall expression
Expects: List { rule : Text, attrs : {...} }
-}
extractTargets :: D.Expr D.Src D.Void -> Either Text [Target]
extractTargets (D.ListLit _ items) = traverse extractTarget (toList items)
  where
    toList = foldr (:) []
extractTargets other = Left $ "Expected list of targets, got: " <> T.pack (show other)

extractTarget :: D.Expr D.Src D.Void -> Either Text Target
extractTarget (D.RecordLit fields) = do
    rule <- case DM.lookup "rule" fields of
        Just (D.RecordField _ (D.TextLit (D.Chunks [] r)) _ _) -> parseRule r
        _ -> Left "Missing or invalid 'rule' field"

    attrs <- case DM.lookup "attrs" fields of
        Just (D.RecordField _ a _ _) -> Right a
        _ -> Left "Missing 'attrs' field"

    name <- case attrs of
        D.RecordLit fs -> case DM.lookup "common" fs of
            Just (D.RecordField _ (D.RecordLit common) _ _) ->
                case DM.lookup "name" common of
                    Just (D.RecordField _ (D.TextLit (D.Chunks [] n)) _ _) -> Right n
                    _ -> Left "Missing name in common attrs"
            _ -> Left "Missing common attrs"
        _ -> Left "attrs is not a record"

    return Target{targetRule = rule, targetName = name, targetAttrs = attrs}
extractTarget other = Left $ "Expected record, got: " <> T.pack (show other)

parseRule :: Text -> Either Text RuleKind
parseRule = \case
    "rust_library" -> Right RustLibrary
    "rust_binary" -> Right RustBinary
    "rust_test" -> Right RustTest
    "lean_library" -> Right LeanLibrary
    "lean_binary" -> Right LeanBinary
    "lean_test" -> Right LeanTest
    "proven_library" -> Right ProvenLibrary
    "cxx_library" -> Right CxxLibrary
    "cxx_binary" -> Right CxxBinary
    "cxx_test" -> Right CxxTest
    "haskell_library" -> Right HaskellLibrary
    "haskell_binary" -> Right HaskellBinary
    "haskell_test" -> Right HaskellTest
    "purescript_library" -> Right PureScriptLibrary
    "purescript_bundle" -> Right PureScriptBundle
    "purescript_test" -> Right PureScriptTest
    "nv_library" -> Right NvLibrary
    "nv_binary" -> Right NvBinary
    "nv_test" -> Right NvTest
    "alias" -> Right Alias
    "filegroup" -> Right Filegroup
    other -> Left $ "Unknown rule: " <> other

-- =============================================================================
-- CLI
-- =============================================================================

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--help"] -> putStrLn usage
        [input] -> transpileFile input "buck"
        [input, "--nix"] -> transpileFile input "nix"
        [input, "--buck"] -> transpileFile input "buck"
        _ -> putStrLn usage

usage :: String
usage =
    unlines
        [ "straylight-transpile: Dhall → Starlark/Nix"
        , ""
        , "Usage: straylight-transpile BUILD.dhall [--buck|--nix]"
        , ""
        , "  --buck  Output BUCK file (default)"
        , "  --nix   Output Nix expression"
        ]

transpileFile :: FilePath -> String -> IO ()
transpileFile input format = do
    -- Parse and normalize Dhall
    expr <- Dhall.inputExpr (T.pack input)

    case extractTargets expr of
        Left err -> error $ T.unpack err
        Right targets -> do
            let output = case format of
                    "nix" -> T.unlines $ map (toNix . targetAttrs) targets
                    "buck" -> toBuck targets
                    _ -> toBuck targets

            let outFile = case format of
                    "nix" -> input -<.> "nix"
                    "buck" -> takeDirectory input </> "BUCK"
                    _ -> takeDirectory input </> "BUCK"

            TIO.writeFile outFile output
            putStrLn $ "Wrote " ++ outFile
