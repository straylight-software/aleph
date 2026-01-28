{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : Render.Nix.Parse
-- Description : Parse Nix files to extract embedded bash scripts
--
-- Uses hnix to parse Nix expressions and find writeShellScript* calls.
-- Extracts the bash string content along with Nix interpolation sites.
--
-- Key patterns we look for:
--
--   pkgs.writeShellScript "name" ''
--     bash content with ${pkgs.foo}/bin/bar
--   ''
--
--   pkgs.writeShellScriptBin "name" ''...''
--
--   lib.writeScript "name" ''...''
--
-- Each interpolation ${...} is tracked with its source span and
-- the Nix expression it contains (for store path verification).
module Render.Nix.Parse
  ( -- * Parsing
    parseNixFile,
    parseNixExpr,

    -- * Extraction
    extractBashScripts,
    BashScript (..),
    Interpolation (..),

    -- * Low-level
    findShellScriptCalls,
    ShellScriptCall (..),
  )
where

import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List.NonEmpty (NonEmpty (..))
import Data.Maybe (mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import Render.Types (Loc (..), Span (..))
import Text.Megaparsec.Pos (unPos)

-- | A bash script extracted from a Nix file
data BashScript = BashScript
  { bsName :: !Text, -- Script name (from first argument)
    bsContent :: !Text, -- Raw bash content with interpolations as placeholders
    bsInterpolations :: ![Interpolation], -- All interpolation sites
    bsSpan :: !Span -- Source location of the bash string
  }
  deriving (Eq, Show)

-- | An interpolation site in a bash string
data Interpolation = Interpolation
  { intExpr :: !Text, -- The Nix expression text (e.g., "pkgs.curl")
    intIsStorePath :: !Bool, -- True if this looks like a store path access
    intSpan :: !Span -- Source location within the bash string
  }
  deriving (Eq, Show)

-- | A writeShellScript* call found in Nix
data ShellScriptCall = ShellScriptCall
  { sscFunction :: !Text, -- "writeShellScript", "writeShellScriptBin", etc.
    sscName :: !Text, -- Script name
    sscBody :: !NExprLoc, -- The body expression (usually a string)
    sscSpan :: !Span -- Source location of the call
  }
  deriving (Show)

-- | Parse a Nix file and return the annotated AST
parseNixFile :: FilePath -> IO (Either Text NExprLoc)
parseNixFile path = do
  result <- parseNixFileLoc (Nix.Path path)
  pure $ case result of
    Left doc -> Left (T.pack $ show doc)
    Right expr -> Right expr

-- | Parse a Nix expression from text
parseNixExpr :: Text -> Either Text NExprLoc
parseNixExpr src = case parseNixTextLoc src of
  Left doc -> Left (T.pack $ show doc)
  Right expr -> Right expr

-- | Extract all bash scripts from a Nix file
extractBashScripts :: FilePath -> IO (Either Text [BashScript])
extractBashScripts path = do
  result <- parseNixFile path
  case result of
    Left err -> pure $ Left err
    Right expr -> pure $ Right $ concatMap extractFromCall (findShellScriptCalls expr)

-- | Extract bash content from a shell script call
extractFromCall :: ShellScriptCall -> [BashScript]
extractFromCall ssc = case extractString (sscBody ssc) of
  Nothing -> []
  Just (content, interps, span') ->
    [ BashScript
        { bsName = sscName ssc,
          bsContent = content,
          bsInterpolations = interps,
          bsSpan = span'
        }
    ]

-- | Extract string content and interpolations from an expression
extractString :: NExprLoc -> Maybe (Text, [Interpolation], Span)
extractString (Fix (Compose (AnnUnit srcSpan expr))) = case expr of
  NStr (DoubleQuoted parts) -> Just (extractParts parts, extractInterps parts, toSpan srcSpan Nothing)
  NStr (Indented _ parts) -> Just (extractParts parts, extractInterps parts, toSpan srcSpan Nothing)
  _ -> Nothing

-- | Extract text from string parts, replacing antiquotes with placeholders
extractParts :: [Antiquoted Text NExprLoc] -> Text
extractParts = T.concat . map extractPart
  where
    extractPart (Plain t) = t
    extractPart EscapedNewline = "\n"
    extractPart (Antiquoted _) = "${...}" -- Placeholder for interpolation

-- | Extract interpolations from string parts
extractInterps :: [Antiquoted Text NExprLoc] -> [Interpolation]
extractInterps = mapMaybe extractInterp
  where
    extractInterp (Plain _) = Nothing
    extractInterp EscapedNewline = Nothing
    extractInterp (Antiquoted expr) =
      Just
        Interpolation
          { intExpr = prettyExpr expr,
            intIsStorePath = isStorePathExpr expr,
            intSpan = exprSpan expr
          }

-- | Check if an expression looks like a store path access
-- e.g., ${pkgs.curl} or ${lib.getExe pkgs.ripgrep}
isStorePathExpr :: NExprLoc -> Bool
isStorePathExpr (Fix (Compose (AnnUnit _ expr))) = case expr of
  -- pkgs.foo or lib.foo
  NSelect _ base (k :| _) -> isPackageBase base || keyTextIs "pkgs" k || keyTextIs "lib" k
  -- lib.getExe pkgs.foo
  NApp func arg -> isStorePathExpr func || isStorePathExpr arg
  -- Direct reference like ${myPkg}
  NSym name -> isLikelyPackageVar (varNameText name)
  -- Literal path /nix/store/...
  NLiteralPath p -> "/nix/store" `T.isPrefixOf` T.pack (show p)
  _ -> False
  where
    isPackageBase (Fix (Compose (AnnUnit _ (NSym n)))) = varNameText n `elem` ["pkgs", "lib"]
    isPackageBase (Fix (Compose (AnnUnit _ (NSelect _ b _ )))) = isPackageBase b
    isPackageBase _ = False

    keyTextIs name (StaticKey k) = varNameText k == name
    keyTextIs _ (DynamicKey _) = False

    isLikelyPackageVar name =
      T.isPrefixOf "pkgs" name
        || T.isPrefixOf "lib" name
        || T.isSuffixOf "Pkg" name
        || T.isSuffixOf "Package" name

-- | Extract text from VarName newtype
varNameText :: VarName -> Text
varNameText = coerce

-- | Get a simple text representation of an expression
prettyExpr :: NExprLoc -> Text
prettyExpr (Fix (Compose (AnnUnit _ expr))) = case expr of
  NSym name -> varNameText name
  NSelect _ base (attr :| rest) ->
    prettyExpr base <> "." <> T.intercalate "." (map keyText (attr : rest))
  NApp func arg -> prettyExpr func <> " " <> prettyExpr arg
  NConstant (NInt n) -> T.pack (show n)
  NConstant (NFloat f) -> T.pack (show f)
  NConstant (NBool b) -> if b then "true" else "false"
  NConstant NNull -> "null"
  NStr _ -> "<string>"
  NList _ -> "<list>"
  NSet _ _ -> "<attrset>"
  NLiteralPath p -> T.pack (show p)
  NEnvPath p -> "<" <> T.pack (show p) <> ">"
  _ -> "<expr>"
  where
    keyText (StaticKey k) = varNameText k
    keyText (DynamicKey _) = "<dynamic>"

-- | Get the source span of an expression
exprSpan :: NExprLoc -> Span
exprSpan (Fix (Compose (AnnUnit srcSpan _))) = toSpan srcSpan Nothing

-- | Convert hnix source span to our Span type
toSpan :: SrcSpan -> Maybe FilePath -> Span
toSpan srcSpan mFile =
  Span
    { spanStart = Loc (sourceLine $ getSpanBegin srcSpan) (sourceCol $ getSpanBegin srcSpan),
      spanEnd = Loc (sourceLine $ getSpanEnd srcSpan) (sourceCol $ getSpanEnd srcSpan),
      spanFile = mFile
    }
  where
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)

-- | Find all writeShellScript* calls in an expression
findShellScriptCalls :: NExprLoc -> [ShellScriptCall]
findShellScriptCalls = go
  where
    go :: NExprLoc -> [ShellScriptCall]
    go expr@(Fix (Compose (AnnUnit srcSpan e))) = case e of
      -- Function application: check if it's a shell script writer
      NApp _ _ -> case unwrapApp expr [] of
        -- Pattern 1: writeShellScript "name" ''body''
        Just (name, [nameArg, bodyArg])
          | isPositionalShellFunc name ->
              case extractStringLit nameArg of
                Just scriptName ->
                  [ ShellScriptCall
                      { sscFunction = name,
                        sscName = scriptName,
                        sscBody = bodyArg,
                        sscSpan = toSpan srcSpan Nothing
                      }
                  ]
                Nothing -> recurse e
        -- Pattern 2: writeShellApplication { name = "foo"; text = ''body''; }
        Just (name, [recordArg])
          | isRecordShellFunc name ->
              case extractFromRecord recordArg of
                Just (scriptName, body) ->
                  [ ShellScriptCall
                      { sscFunction = name,
                        sscName = scriptName,
                        sscBody = body,
                        sscSpan = toSpan srcSpan Nothing
                      }
                  ]
                Nothing -> recurse e
        _ -> recurse e
      _ -> recurse e

    -- Unwrap nested applications to find the function name and all arguments
    unwrapApp :: NExprLoc -> [NExprLoc] -> Maybe (Text, [NExprLoc])
    unwrapApp (Fix (Compose (AnnUnit _ e))) args = case e of
      NApp func arg -> unwrapApp func (arg : args)
      NSym name -> Just (varNameText name, args)
      NSelect _ _ (attr :| rest) ->
        Just (keyText (last (attr : rest)), args)
      _ -> Nothing

    keyText (StaticKey k) = varNameText k
    keyText (DynamicKey _) = ""

    -- Check if a function takes positional args: writeShellScript "name" ''body''
    isPositionalShellFunc :: Text -> Bool
    isPositionalShellFunc name =
      name == "writeShellScript"
        || name == "writeShellScriptBin"
        || name == "writeScript"
        || name == "writeScriptBin"

    -- Check if a function takes a record arg: writeShellApplication { ... }
    isRecordShellFunc :: Text -> Bool
    isRecordShellFunc name = name == "writeShellApplication"

    -- Extract name and text from a record: { name = "foo"; text = ''body''; }
    extractFromRecord :: NExprLoc -> Maybe (Text, NExprLoc)
    extractFromRecord (Fix (Compose (AnnUnit _ e))) = case e of
      NSet _ bindings ->
        let nameVal = findBinding "name" bindings >>= extractStringLit
            textVal = findBinding "text" bindings
        in case (nameVal, textVal) of
          (Just n, Just t) -> Just (n, t)
          _ -> Nothing
      _ -> Nothing

    -- Find a binding by name in a binding list
    findBinding :: Text -> [Binding NExprLoc] -> Maybe NExprLoc
    findBinding name = foldr check Nothing
      where
        check (NamedVar (StaticKey k :| []) expr _) acc
          | varNameText k == name = Just expr
          | otherwise = acc
        check _ acc = acc

    -- Extract a string literal from an expression
    extractStringLit :: NExprLoc -> Maybe Text
    extractStringLit (Fix (Compose (AnnUnit _ e))) = case e of
      NStr (DoubleQuoted [Plain t]) -> Just t
      NStr (Indented _ [Plain t]) -> Just t
      _ -> Nothing

    -- Recurse into sub-expressions
    recurse :: NExprF NExprLoc -> [ShellScriptCall]
    recurse e = case e of
      NConstant _ -> []
      NStr _ -> []
      NSym _ -> []
      NList xs -> concatMap go xs
      NSet _ bindings -> concatMap goBinding bindings
      NLiteralPath _ -> []
      NEnvPath _ -> []
      NLet bindings body -> concatMap goBinding bindings ++ go body
      NIf cond t f -> go cond ++ go t ++ go f
      NWith scope body -> go scope ++ go body
      NAssert cond body -> go cond ++ go body
      NAbs _ body -> go body
      NApp f x -> go f ++ go x
      NSelect alt base _ -> go base ++ maybe [] go alt
      NHasAttr base _ -> go base
      NUnary _ x -> go x
      NBinary _ x y -> go x ++ go y
      NSynHole _ -> []

    goBinding :: Binding NExprLoc -> [ShellScriptCall]
    goBinding = \case
      NamedVar _ expr _ -> go expr
      Inherit _ _ _ -> []
