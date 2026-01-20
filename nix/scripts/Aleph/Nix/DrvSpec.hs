{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | CA-Derivation types matching Aleph/Config/Drv.dhall
  
These types are designed for bidirectional conversion with Dhall:
  - ToDhall: write Haskell specs to Dhall (drvToDhall)
  - toNix: convert to Nix Value for WASM boundary

DHALL IS THE SUBSTRATE.

The types mirror the Dhall schema exactly, enabling:
  1. Type-safe derivation construction in Haskell
  2. Dhall as the serialization format between eval and build
  3. WASM plugins that understand the same schema

= Zero-Bash Architecture (RFC-007)

The key function is 'drvToDhall' which emits self-contained Dhall text:

@
-- In your package definition:
pkg :: DrvSpec
pkg = defaultDrvSpec { pname = "foo", ... }

-- Emit Dhall (with placeholders for Nix to resolve):
dhallText = drvToDhall pkg
-- Returns: "{ pname = \"foo\", src = @src@, ... }"
@

Placeholders:
  - @src@ → resolved source path (Nix substitutes this)
  - @dep:name@ → resolved dependency path
  - @out@ → output path ($out at build time)
-}
module Aleph.Nix.DrvSpec (
    -- * Hash types
    HashAlgo (..),
    Hash (..),
    SriHash,
    
    -- * Store primitives
    StorePath (..),
    DrvPath (..),
    OutputMethod (..),
    
    -- * References
    Ref (..),
    dep, depSub, out, outSub, outNamed, src, srcSub,
    
    -- * Build actions
    Mode (..),
    Cmp (..),
    StreamTarget (..),
    Expr (..),
    Compression (..),
    Generator (..),
    LogLevel (..),
    Action (..),
    
    -- * Dependencies
    DepKind (..),
    Dep (..),
    buildDep, hostDep, checkDep,
    
    -- * Source
    Src (..),
    GitHubSrc (..), 
    UrlSrc (..),
    GitSrc (..),
    
    -- * Outputs
    Output (..),
    floatingOut, fixedOut,
    
    -- * Phases
    Phases (..),
    emptyPhases,
    
    -- * Metadata
    Meta (..),
    
    -- * The derivation
    DrvSpec (..),
    defaultDrvSpec,
    
    -- * Shell hooks
    ShellHooks (..),
    
    -- * Dhall emission (RFC-007 Zero-Bash)
    drvToDhall,
    actionToDhall,
    refToDhall,
    
    -- * Nix emission (WASM boundary)
    drvToNix,
) where

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T

import Aleph.Nix.Value (Value, mkAttrs, mkString, mkBool, mkList, mkNull, mkInt)

-- ============================================================================
-- Hash Types
-- ============================================================================

data HashAlgo = SHA256 | SHA512 | Blake3
    deriving (Eq, Show)

data Hash = Hash
    { algo :: !HashAlgo
    , value :: !Text
    }
    deriving (Eq, Show)

type SriHash = Text

-- ============================================================================
-- Store Primitives
-- ============================================================================

newtype StorePath = StorePath { unStorePath :: Text }
    deriving (Eq, Show)

newtype DrvPath = DrvPath { unDrvPath :: Text }
    deriving (Eq, Show)

data OutputMethod
    = Fixed !SriHash
    | Floating
    deriving (Eq, Show)

-- ============================================================================
-- References
-- ============================================================================

data Ref
    = RefDep { name :: !Text, subpath :: !(Maybe Text) }
    | RefOut { outName :: !Text, subpath :: !(Maybe Text) }
    | RefSrc { subpath :: !(Maybe Text) }
    | RefEnv !Text
    | RefRel !Text
    | RefLit !Text
    | RefCat ![Ref]
    deriving (Eq, Show)


-- Smart constructors
dep :: Text -> Ref
dep n = RefDep n Nothing

depSub :: Text -> Text -> Ref
depSub n s = RefDep n (Just s)

out :: Ref
out = RefOut "out" Nothing

outSub :: Text -> Ref
outSub s = RefOut "out" (Just s)

outNamed :: Text -> Ref
outNamed n = RefOut n Nothing

src :: Ref
src = RefSrc Nothing

srcSub :: Text -> Ref
srcSub s = RefSrc (Just s)

-- ============================================================================
-- Build Actions
-- ============================================================================

data Mode
    = ModeR
    | ModeRW
    | ModeRX
    | ModeRWX
    | ModeOctal !Int
    deriving (Eq, Show)


data Cmp = Eq | Ne | Lt | Le | Gt | Ge
    deriving (Eq, Show)


data StreamTarget
    = Stdout
    | Stderr
    | ToFile !Ref
    | ToNull
    deriving (Eq, Show)


data Expr
    = ExprStr !Text
    | ExprInt !Integer
    | ExprBool !Bool
    | ExprRef !Ref
    | ExprEnv !Text
    | ExprConcat ![Expr]
    | ExprPathExists !Ref
    | ExprFileContents !Ref
    | ExprCompare !Cmp !Expr !Expr
    | ExprAnd !Expr !Expr
    | ExprOr !Expr !Expr
    | ExprNot !Expr
    deriving (Eq, Show)


data Compression = NoCompression | Gzip | Zstd | Xz
    deriving (Eq, Show)


data Generator = Ninja | Make | DefaultGenerator
    deriving (Eq, Show)


data LogLevel = Debug | Info | Warn | Error
    deriving (Eq, Show)


data Action
    -- Filesystem
    = Copy { copySrc :: !Ref, copyDst :: !Ref }
    | Move { moveSrc :: !Ref, moveDst :: !Ref }
    | Symlink { symlinkTarget :: !Ref, symlinkLink :: !Ref }
    | Mkdir { mkdirPath :: !Ref, mkdirParents :: !Bool }
    | Remove { removePath :: !Ref, removeRecursive :: !Bool }
    | Touch !Ref
    | Chmod { chmodPath :: !Ref, chmodMode :: !Mode }
    
    -- File I/O
    | Write { writePath :: !Ref, writeContents :: !Text }
    | Append { appendPath :: !Ref, appendContents :: !Text }
    
    -- Archives
    | Untar { untarSrc :: !Ref, untarDst :: !Ref, untarStrip :: !Int }
    | Unzip { unzipSrc :: !Ref, unzipDst :: !Ref }
    | Tar { tarSrc :: !Ref, tarDst :: !Ref, tarCompression :: !Compression }
    
    -- Patching
    | Patch { patchFile :: !Ref, patchDir :: !Ref, patchStrip :: !Int }
    | Substitute { substFile :: !Ref, substReplacements :: ![(Text, Text)] }
    | SubstituteRef { substRefFile :: !Ref, substRefReplacements :: ![(Text, Ref)] }
    
    -- ELF manipulation
    | PatchElfRpath { patchElfPath :: !Ref, patchElfRpaths :: ![Ref] }
    | PatchElfAddRpath { patchElfAddPath :: !Ref, patchElfAddRpaths :: ![Ref] }
    | PatchElfInterpreter { patchElfInterpPath :: !Ref, patchElfInterpreter :: !Ref }
    | PatchElfShrink { patchElfShrinkPath :: !Ref }
    
    -- Execution
    | Run 
        { runCmd :: !Ref
        , runArgs :: ![Expr]
        , runEnv :: ![(Text, Expr)]
        , runCwd :: !(Maybe Ref)
        , runStdin :: !(Maybe Ref)
        , runStdout :: !StreamTarget
        , runStderr :: !StreamTarget
        }
    | Tool
        { toolDep :: !Text
        , toolBin :: !Text
        , toolArgs :: ![Expr]
        }
    
    -- Build systems
    | CMakeConfigure
        { cmakeSrcDir :: !Ref
        , cmakeBuildDir :: !Ref
        , cmakeInstallPrefix :: !Ref
        , cmakeBuildType :: !Text
        , cmakeFlags :: ![Text]
        , cmakeGenerator :: !Generator
        }
    | CMakeBuild
        { cmakeBuildBuildDir :: !Ref
        , cmakeBuildTarget :: !(Maybe Text)
        , cmakeBuildJobs :: !(Maybe Int)
        }
    | CMakeInstall { cmakeInstallBuildDir :: !Ref }
    
    | MakeAction
        { makeTargets :: ![Text]
        , makeFlags :: ![Text]
        , makeJobs :: !(Maybe Int)
        , makeDir :: !(Maybe Ref)
        }
    
    | MesonSetup
        { mesonSrcDir :: !Ref
        , mesonBuildDir :: !Ref
        , mesonPrefix :: !Ref
        , mesonBuildType :: !Text
        , mesonFlags :: ![Text]
        }
    | NinjaBuild
        { ninjaBuildDir :: !Ref
        , ninjaTargets :: ![Text]
        , ninjaJobs :: !(Maybe Int)
        }
    
    | Configure { configureFlags :: ![Text] }
    
    -- Install helpers
    | InstallBin { installBinSrc :: !Ref }
    | InstallLib { installLibSrc :: !Ref }
    | InstallInclude { installIncludeSrc :: !Ref }
    | InstallShare { installShareSrc :: !Ref, installShareSubdir :: !Text }
    | InstallPkgConfig { installPkgConfigSrc :: !Ref }
    
    -- Control flow
    | If { ifCond :: !Expr, ifThen :: ![Action], ifElse :: ![Action] }
    | ForFiles { forPattern :: !Text, forDir :: !Ref, forVar :: !Text, forDo :: ![Action] }
    | Seq ![Action]
    | Parallel ![Action]
    | Try { tryActions :: ![Action], tryCatch :: ![Action] }
    
    -- Assertions
    | Assert { assertCond :: !Expr, assertMsg :: !Text }
    | Log { logLevel :: !LogLevel, logMsg :: !Text }
    
    -- Escape hatch
    | Shell !Text
    deriving (Eq, Show)


-- ============================================================================
-- Dependencies
-- ============================================================================

data DepKind = Build | Host | Propagate | Check | Data
    deriving (Eq, Show)


data Dep = Dep
    { depName :: !Text
    , depStorePath :: !(Maybe StorePath)
    , depKind :: !DepKind
    , depOutputs :: ![Text]
    }
    deriving (Eq, Show)


buildDep :: Text -> Dep
buildDep n = Dep n Nothing Build ["out"]

hostDep :: Text -> Dep
hostDep n = Dep n Nothing Host ["out"]

checkDep :: Text -> Dep
checkDep n = Dep n Nothing Check ["out"]

-- ============================================================================
-- Source
-- ============================================================================

data GitHubSrc = GitHubSrc
    { ghOwner :: !Text
    , ghRepo :: !Text
    , ghRev :: !Text
    , ghHash :: !SriHash
    }
    deriving (Eq, Show)


data UrlSrc = UrlSrc
    { urlUrl :: !Text
    , urlHash :: !SriHash
    }
    deriving (Eq, Show)


data GitSrc = GitSrc
    { gitUrl :: !Text
    , gitRev :: !Text
    , gitHash :: !SriHash
    }
    deriving (Eq, Show)


data Src
    = SrcGitHub !GitHubSrc
    | SrcGitLab !GitHubSrc  -- same structure
    | SrcUrl !UrlSrc
    | SrcGit !GitSrc
    | SrcStore !StorePath
    | SrcNone
    deriving (Eq, Show)


-- ============================================================================
-- Outputs
-- ============================================================================

data Output = Output
    { outputName :: !Text
    , outputMethod :: !OutputMethod
    }
    deriving (Eq, Show)


floatingOut :: Text -> Output
floatingOut n = Output n Floating

fixedOut :: Text -> SriHash -> Output
fixedOut n h = Output n (Fixed h)

-- ============================================================================
-- Phases
-- ============================================================================

data Phases = Phases
    { unpack :: ![Action]
    , patch :: ![Action]
    , configure :: ![Action]
    , build :: ![Action]
    , check :: ![Action]
    , install :: ![Action]
    , fixup :: ![Action]
    }
    deriving (Eq, Show)


emptyPhases :: Phases
emptyPhases = Phases [] [] [] [] [] [] []

-- ============================================================================
-- Metadata
-- ============================================================================

data Meta = Meta
    { description :: !Text
    , homepage :: !(Maybe Text)
    , license :: !Text
    , maintainers :: ![Text]
    , platforms :: ![Text]
    }
    deriving (Eq, Show)


-- ============================================================================
-- Shell Hooks (escape hatch)
-- ============================================================================

data ShellHooks = ShellHooks
    { preBuild :: !(Maybe Text)
    , postBuild :: !(Maybe Text)
    , preInstall :: !(Maybe Text)
    , postInstall :: !(Maybe Text)
    }
    deriving (Eq, Show)


-- ============================================================================
-- The Derivation Spec
-- ============================================================================

data DrvSpec = DrvSpec
    { pname :: !Text
    , version :: !Text
    , system :: !Text
    , contentAddressed :: !Bool
    , outputs :: ![Output]
    , specSrc :: !Src
    , deps :: ![Dep]
    , phases :: !Phases
    , env :: ![(Text, Text)]
    , meta :: !Meta
    , passthru :: ![(Text, Text)]
    , shellHooks :: !ShellHooks
    }
    deriving (Eq, Show)


defaultDrvSpec :: DrvSpec
defaultDrvSpec = DrvSpec
    { pname = "unnamed"
    , version = "0.0.0"
    , system = "x86_64-linux"
    , contentAddressed = True
    , outputs = [floatingOut "out"]
    , specSrc = SrcNone
    , deps = []
    , phases = emptyPhases
    , env = []
    , meta = Meta "" Nothing "unfree" [] []
    , passthru = []
    , shellHooks = ShellHooks Nothing Nothing Nothing Nothing
    }

-- ============================================================================
-- Dhall Emission (RFC-007 Zero-Bash Architecture)
-- ============================================================================
-- 
-- DHALL IS THE SUBSTRATE.
--
-- These functions emit self-contained Dhall text that can be:
--   1. Written to the Nix store
--   2. Read by aleph-exec at build time
--   3. Executed as typed actions (no shell)
--
-- Placeholders are used for values that Nix must resolve:
--   @src@      → resolved source store path
--   @dep:name@ → resolved dependency store path
--   @out@      → output path (env var at build time)
--

-- | Emit a complete DrvSpec as Dhall text
drvToDhall :: DrvSpec -> Text
drvToDhall DrvSpec{..} = T.unlines
    [ dhallTypes
    , ""
    , "in"
    , "  { pname = " <> dhallText pname
    , "  , version = " <> dhallText version
    , "  , system = " <> dhallText system
    , "  , src = " <> srcToDhall specSrc
    , "  , phases ="
    , "      { unpack = " <> actionsToDhall (unpack phases)
    , "      , patch = " <> actionsToDhall (patch phases)
    , "      , configure = " <> actionsToDhall (configure phases)
    , "      , build = " <> actionsToDhall (build phases)
    , "      , check = " <> actionsToDhall (check phases)
    , "      , install = " <> actionsToDhall (install phases)
    , "      , fixup = " <> actionsToDhall (fixup phases)
    , "      }"
    , "  , meta ="
    , "      { description = " <> dhallText (description meta)
    , "      , homepage = " <> dhallMaybe dhallText (homepage meta)
    , "      , license = " <> dhallText (license meta)
    , "      , maintainers = " <> dhallList dhallText (maintainers meta)
    , "      , platforms = " <> dhallList dhallText (platforms meta)
    , "      }"
    , "  }"
    ]

-- | Dhall type definitions (inlined, no imports)
dhallTypes :: Text
dhallTypes = T.unlines
    [ "let Ref ="
    , "  < Dep : { name : Text, subpath : Optional Text }"
    , "  | Out : { name : Text, subpath : Optional Text }"
    , "  | Src : { subpath : Optional Text }"
    , "  | Env : Text"
    , "  | Rel : Text"
    , "  | Lit : Text"
    , "  >"
    , ""
    , "let Replacement = { from : Text, to : Text }"
    , ""
    , "let Mode = < R | RW | RX | RWX | Octal : Natural >"
    , ""
    , "let Generator = < Ninja | Make | Default >"
    , ""
    , "let Action ="
    , "  < Copy : { src : Ref, dst : Ref }"
    , "  | Move : { src : Ref, dst : Ref }"
    , "  | Symlink : { target : Ref, link : Ref }"
    , "  | Mkdir : { path : Ref, parents : Bool }"
    , "  | Remove : { path : Ref, recursive : Bool }"
    , "  | Touch : Ref"
    , "  | Chmod : { path : Ref, mode : Mode }"
    , "  | Write : { path : Ref, contents : Text }"
    , "  | Append : { path : Ref, contents : Text }"
    , "  | Untar : { src : Ref, dst : Ref, strip : Natural }"
    , "  | Unzip : { src : Ref, dst : Ref }"
    , "  | Substitute : { file : Ref, replacements : List Replacement }"
    , "  | PatchElfRpath : { path : Ref, rpaths : List Ref }"
    , "  | PatchElfInterpreter : { path : Ref, interpreter : Ref }"
    , "  | PatchElfShrink : { path : Ref }"
    , "  | CMake : { srcDir : Ref, buildDir : Ref, installPrefix : Ref, buildType : Text, flags : List Text, generator : Generator }"
    , "  | CMakeBuild : { buildDir : Ref, target : Optional Text, jobs : Optional Natural }"
    , "  | CMakeInstall : { buildDir : Ref }"
    , "  | Make : { targets : List Text, flags : List Text, jobs : Optional Natural, dir : Optional Ref }"
    , "  | InstallBin : { src : Ref }"
    , "  | InstallLib : { src : Ref }"
    , "  | InstallInclude : { src : Ref }"
    , "  | Shell : Text"
    , "  >"
    , ""
    , "let Phases ="
    , "  { unpack : List Action"
    , "  , patch : List Action"
    , "  , configure : List Action"
    , "  , build : List Action"
    , "  , check : List Action"
    , "  , install : List Action"
    , "  , fixup : List Action"
    , "  }"
    , ""
    , "let Src ="
    , "  < GitHub : { owner : Text, repo : Text, rev : Text, hash : Text }"
    , "  | Url : { url : Text, hash : Text }"
    , "  | Store : Text"
    , "  | None"
    , "  >"
    , ""
    , "let Meta ="
    , "  { description : Text"
    , "  , homepage : Optional Text"
    , "  , license : Text"
    , "  , maintainers : List Text"
    , "  , platforms : List Text"
    , "  }"
    ]

-- | Convert Ref to Dhall
refToDhall :: Ref -> Text
refToDhall = \case
    RefDep n msub -> "Ref.Dep { name = " <> dhallText n <> ", subpath = " <> dhallMaybe dhallText msub <> " }"
    RefOut n msub -> "Ref.Out { name = " <> dhallText n <> ", subpath = " <> dhallMaybe dhallText msub <> " }"
    RefSrc msub -> "Ref.Src { subpath = " <> dhallMaybe dhallText msub <> " }"
    RefEnv v -> "Ref.Env " <> dhallText v
    RefRel p -> "Ref.Rel " <> dhallText p
    RefLit t -> "Ref.Lit " <> dhallText t
    RefCat refs -> "Ref.Cat " <> dhallList refToDhall refs

-- | Convert Action to Dhall
actionToDhall :: Action -> Text
actionToDhall = \case
    Copy s d -> 
        "Action.Copy { src = " <> refToDhall s <> ", dst = " <> refToDhall d <> " }"
    Move s d -> 
        "Action.Move { src = " <> refToDhall s <> ", dst = " <> refToDhall d <> " }"
    Symlink t l -> 
        "Action.Symlink { target = " <> refToDhall t <> ", link = " <> refToDhall l <> " }"
    Mkdir p parents -> 
        "Action.Mkdir { path = " <> refToDhall p <> ", parents = " <> dhallBool parents <> " }"
    Remove p recursive -> 
        "Action.Remove { path = " <> refToDhall p <> ", recursive = " <> dhallBool recursive <> " }"
    Touch p -> 
        "Action.Touch " <> refToDhall p
    Chmod p m -> 
        "Action.Chmod { path = " <> refToDhall p <> ", mode = " <> modeToDhall m <> " }"
    Write p c -> 
        "Action.Write { path = " <> refToDhall p <> ", contents = " <> dhallText c <> " }"
    Append p c -> 
        "Action.Append { path = " <> refToDhall p <> ", contents = " <> dhallText c <> " }"
    Untar s d strip -> 
        "Action.Untar { src = " <> refToDhall s <> ", dst = " <> refToDhall d <> ", strip = " <> T.pack (show strip) <> " }"
    Unzip s d -> 
        "Action.Unzip { src = " <> refToDhall s <> ", dst = " <> refToDhall d <> " }"
    Tar s d comp ->
        "Action.Tar { src = " <> refToDhall s <> ", dst = " <> refToDhall d <> ", compression = " <> compressionToDhall comp <> " }"
    Patch f d strip ->
        "Action.Patch { patch = " <> refToDhall f <> ", dir = " <> refToDhall d <> ", strip = " <> T.pack (show strip) <> " }"
    Substitute f reps -> 
        "Action.Substitute { file = " <> refToDhall f <> ", replacements = " <> dhallList repToDhall reps <> " }"
      where repToDhall (from, to) = "{ from = " <> dhallText from <> ", to = " <> dhallText to <> " }"
    SubstituteRef f reps ->
        "Action.SubstituteRef { file = " <> refToDhall f <> ", replacements = " <> dhallList repToDhall reps <> " }"
      where repToDhall (from, to) = "{ from = " <> dhallText from <> ", to = " <> refToDhall to <> " }"
    PatchElfRpath p rpaths -> 
        "Action.PatchElfRpath { path = " <> refToDhall p <> ", rpaths = " <> dhallList refToDhall rpaths <> " }"
    PatchElfAddRpath p rpaths ->
        "Action.PatchElfAddRpath { path = " <> refToDhall p <> ", rpaths = " <> dhallList refToDhall rpaths <> " }"
    PatchElfInterpreter p i -> 
        "Action.PatchElfInterpreter { path = " <> refToDhall p <> ", interpreter = " <> refToDhall i <> " }"
    PatchElfShrink p -> 
        "Action.PatchElfShrink { path = " <> refToDhall p <> " }"
    Run cmd args envVars cwd stdin stdout stderr ->
        "Action.Run { cmd = " <> refToDhall cmd 
        <> ", args = " <> dhallList exprToDhall args
        <> ", env = " <> dhallList (\(k,v) -> "{ key = " <> dhallText k <> ", value = " <> exprToDhall v <> " }") envVars
        <> ", cwd = " <> dhallMaybe refToDhall cwd
        <> ", stdin = " <> dhallMaybe refToDhall stdin
        <> ", stdout = " <> streamTargetToDhall stdout
        <> ", stderr = " <> streamTargetToDhall stderr
        <> " }"
    Tool dep bin args ->
        "Action.Tool { dep = " <> dhallText dep <> ", bin = " <> dhallText bin <> ", args = " <> dhallList exprToDhall args <> " }"
    CMakeConfigure srcDir buildDir prefix buildType flags gen ->
        "Action.CMake { srcDir = " <> refToDhall srcDir
        <> ", buildDir = " <> refToDhall buildDir
        <> ", installPrefix = " <> refToDhall prefix
        <> ", buildType = " <> dhallText buildType
        <> ", flags = " <> dhallList dhallText flags
        <> ", generator = " <> generatorToDhall gen
        <> " }"
    CMakeBuild buildDir target jobs ->
        "Action.CMakeBuild { buildDir = " <> refToDhall buildDir
        <> ", target = " <> dhallMaybe dhallText target
        <> ", jobs = " <> dhallMaybe (T.pack . show) jobs
        <> " }"
    CMakeInstall buildDir ->
        "Action.CMakeInstall { buildDir = " <> refToDhall buildDir <> " }"
    MakeAction targets flags jobs dir ->
        "Action.Make { targets = " <> dhallList dhallText targets
        <> ", flags = " <> dhallList dhallText flags
        <> ", jobs = " <> dhallMaybe (T.pack . show) jobs
        <> ", dir = " <> dhallMaybe refToDhall dir
        <> " }"
    MesonSetup srcDir buildDir prefix buildType flags ->
        "Action.Meson { srcDir = " <> refToDhall srcDir
        <> ", buildDir = " <> refToDhall buildDir
        <> ", prefix = " <> refToDhall prefix
        <> ", buildType = " <> dhallText buildType
        <> ", flags = " <> dhallList dhallText flags
        <> " }"
    NinjaBuild buildDir targets jobs ->
        "Action.NinjaBuild { buildDir = " <> refToDhall buildDir
        <> ", targets = " <> dhallList dhallText targets
        <> ", jobs = " <> dhallMaybe (T.pack . show) jobs
        <> " }"
    Configure flags ->
        "Action.Configure { flags = " <> dhallList dhallText flags <> " }"
    InstallBin s -> 
        "Action.InstallBin { src = " <> refToDhall s <> " }"
    InstallLib s -> 
        "Action.InstallLib { src = " <> refToDhall s <> " }"
    InstallInclude s -> 
        "Action.InstallInclude { src = " <> refToDhall s <> " }"
    InstallShare s subdir ->
        "Action.InstallShare { src = " <> refToDhall s <> ", subdir = " <> dhallText subdir <> " }"
    InstallPkgConfig s ->
        "Action.InstallPkgConfig { src = " <> refToDhall s <> " }"
    If cond then_ else_ ->
        "Action.If { cond = " <> exprToDhall cond
        <> ", then_ = " <> dhallList actionToDhall then_
        <> ", else_ = " <> dhallList actionToDhall else_
        <> " }"
    ForFiles pat dir var actions ->
        "Action.ForFiles { pattern = " <> dhallText pat
        <> ", dir = " <> refToDhall dir
        <> ", var = " <> dhallText var
        <> ", do = " <> dhallList actionToDhall actions
        <> " }"
    Seq actions -> 
        "Action.Seq " <> dhallList actionToDhall actions
    Parallel actions ->
        "Action.Parallel " <> dhallList actionToDhall actions
    Try actions catch ->
        "Action.Try { actions = " <> dhallList actionToDhall actions
        <> ", catch = " <> dhallList actionToDhall catch
        <> " }"
    Assert cond msg ->
        "Action.Assert { cond = " <> exprToDhall cond <> ", msg = " <> dhallText msg <> " }"
    Log level msg ->
        "Action.Log { level = " <> logLevelToDhall level <> ", msg = " <> dhallText msg <> " }"
    Shell cmd -> 
        "Action.Shell " <> dhallText cmd

-- | Convert list of actions to Dhall
actionsToDhall :: [Action] -> Text
actionsToDhall [] = "[] : List Action"
actionsToDhall as = dhallList actionToDhall as

-- | Convert Src to Dhall
srcToDhall :: Src -> Text
srcToDhall = \case
    SrcGitHub gh -> "Src.GitHub { owner = " <> dhallText (ghOwner gh) 
                    <> ", repo = " <> dhallText (ghRepo gh)
                    <> ", rev = " <> dhallText (ghRev gh)
                    <> ", hash = " <> dhallText (ghHash gh) <> " }"
    SrcGitLab gl -> "Src.GitLab { owner = " <> dhallText (ghOwner gl)
                    <> ", repo = " <> dhallText (ghRepo gl)
                    <> ", rev = " <> dhallText (ghRev gl)
                    <> ", hash = " <> dhallText (ghHash gl) <> " }"
    SrcUrl u -> "Src.Url { url = " <> dhallText (urlUrl u) <> ", hash = " <> dhallText (urlHash u) <> " }"
    SrcGit g -> "Src.Git { url = " <> dhallText (gitUrl g) <> ", rev = " <> dhallText (gitRev g) <> ", hash = " <> dhallText (gitHash g) <> " }"
    SrcStore (StorePath p) -> "Src.Store " <> dhallText p
    SrcNone -> "Src.None"

-- | Convert Mode to Dhall
modeToDhall :: Mode -> Text
modeToDhall = \case
    ModeR -> "Mode.R"
    ModeRW -> "Mode.RW"
    ModeRX -> "Mode.RX"
    ModeRWX -> "Mode.RWX"
    ModeOctal n -> "Mode.Octal " <> T.pack (show n)

-- | Convert Generator to Dhall
generatorToDhall :: Generator -> Text
generatorToDhall = \case
    Ninja -> "Generator.Ninja"
    Make -> "Generator.Make"
    DefaultGenerator -> "Generator.Default"

-- | Convert Compression to Dhall
compressionToDhall :: Compression -> Text
compressionToDhall = \case
    NoCompression -> "< None | Gzip | Zstd | Xz >.None"
    Gzip -> "< None | Gzip | Zstd | Xz >.Gzip"
    Zstd -> "< None | Gzip | Zstd | Xz >.Zstd"
    Xz -> "< None | Gzip | Zstd | Xz >.Xz"

-- | Convert Expr to Dhall
exprToDhall :: Expr -> Text
exprToDhall = \case
    ExprStr t -> "Expr.Str " <> dhallText t
    ExprInt n -> "Expr.Int " <> T.pack (show n)
    ExprBool b -> "Expr.Bool " <> dhallBool b
    ExprRef r -> "Expr.Ref " <> refToDhall r
    ExprEnv v -> "Expr.Env " <> dhallText v
    ExprConcat es -> "Expr.Concat " <> dhallList exprToDhall es
    ExprPathExists r -> "Expr.PathExists " <> refToDhall r
    ExprFileContents r -> "Expr.FileContents " <> refToDhall r
    ExprCompare op a b -> "Expr.Compare { op = " <> cmpToDhall op <> ", a = " <> exprToDhall a <> ", b = " <> exprToDhall b <> " }"
    ExprAnd a b -> "Expr.And { a = " <> exprToDhall a <> ", b = " <> exprToDhall b <> " }"
    ExprOr a b -> "Expr.Or { a = " <> exprToDhall a <> ", b = " <> exprToDhall b <> " }"
    ExprNot e -> "Expr.Not " <> exprToDhall e

-- | Convert Cmp to Dhall
cmpToDhall :: Cmp -> Text
cmpToDhall = \case
    Eq -> "Cmp.Eq"
    Ne -> "Cmp.Ne"
    Lt -> "Cmp.Lt"
    Le -> "Cmp.Le"
    Gt -> "Cmp.Gt"
    Ge -> "Cmp.Ge"

-- | Convert StreamTarget to Dhall
streamTargetToDhall :: StreamTarget -> Text
streamTargetToDhall = \case
    Stdout -> "StreamTarget.Stdout"
    Stderr -> "StreamTarget.Stderr"
    ToFile r -> "StreamTarget.File " <> refToDhall r
    ToNull -> "StreamTarget.Null"

-- | Convert LogLevel to Dhall
logLevelToDhall :: LogLevel -> Text
logLevelToDhall = \case
    Debug -> "< Debug | Info | Warn | Error >.Debug"
    Info -> "< Debug | Info | Warn | Error >.Info"
    Warn -> "< Debug | Info | Warn | Error >.Warn"
    Error -> "< Debug | Info | Warn | Error >.Error"

-- ============================================================================
-- Dhall Helpers
-- ============================================================================

-- | Escape and quote a Text for Dhall
dhallText :: Text -> Text
dhallText t = "\"" <> escape t <> "\""
  where
    escape = T.concatMap $ \case
        '\\' -> "\\\\"
        '"'  -> "\\\""
        '\n' -> "\\n"
        '\r' -> "\\r"
        '\t' -> "\\t"
        c    -> T.singleton c

-- | Bool to Dhall
dhallBool :: Bool -> Text
dhallBool True = "True"
dhallBool False = "False"

-- | Maybe to Dhall Optional
dhallMaybe :: (a -> Text) -> Maybe a -> Text
dhallMaybe _ Nothing = "None Text"
dhallMaybe f (Just x) = "Some " <> f x

-- | List to Dhall
dhallList :: (a -> Text) -> [a] -> Text
dhallList _ [] = "[] : List _"  -- Type annotation needed for empty lists
dhallList f xs = "[" <> T.intercalate ", " (map f xs) <> "]"

-- ============================================================================
-- Nix Emission (WASM Boundary)
-- ============================================================================
--
-- Convert DrvSpec to Nix Value for the WASM FFI boundary.
-- This is what gets returned to Nix from builtins.wasm calls.
--
-- The returned attrset includes:
--   - All derivation fields (pname, version, src, deps, etc.)
--   - dhall: the Dhall spec for aleph-exec (zero-bash path)
--
-- Nix then either:
--   1. Passes dhall to aleph-exec (zero-bash mode)
--   2. Uses the fields to call stdenv.mkDerivation (legacy mode)
--

-- | Convert DrvSpec to Nix Value for WASM boundary
drvToNix :: DrvSpec -> IO Value
drvToNix drv@DrvSpec{..} = do
    pairs <- sequence
        [ ("pname",) <$> mkString pname
        , ("version",) <$> mkString version
        , ("system",) <$> mkString system
        , ("src",) <$> srcToNix specSrc
        , ("deps",) <$> depsToNix deps
        , ("builder",) <$> builderToNix
        , ("meta",) <$> metaToNix meta
        , ("phases",) <$> phasesToNix phases
        , ("env",) <$> envToNix env
        , ("strictDeps",) <$> mkBool True
        , ("doCheck",) <$> mkBool False
        , ("dontUnpack",) <$> mkBool False
        -- Zero-bash: include Dhall spec for aleph-exec
        , ("dhall",) <$> mkString (drvToDhall drv)
        ]
    mkAttrs (Map.fromList pairs)
  where
    -- Determine builder type from phases
    builderToNix = do
        -- For now, detect cmake from configure phase
        -- TODO: add explicit builder field to DrvSpec
        let hasCMake = any isCMakeAction (configure phases)
        if hasCMake
            then do
                pairs <- sequence
                    [ ("type",) <$> mkString "cmake"
                    , ("flags",) <$> mkList []
                    ]
                mkAttrs (Map.fromList pairs)
            else do
                pairs <- sequence [("type",) <$> mkString "none"]
                mkAttrs (Map.fromList pairs)
    
    isCMakeAction (CMakeConfigure {}) = True
    isCMakeAction _ = False

-- | Convert Src to Nix
srcToNix :: Src -> IO Value
srcToNix = \case
    SrcGitHub GitHubSrc{..} -> do
        pairs <- sequence
            [ ("type",) <$> mkString "github"
            , ("owner",) <$> mkString ghOwner
            , ("repo",) <$> mkString ghRepo
            , ("rev",) <$> mkString ghRev
            , ("hash",) <$> mkString ghHash
            ]
        mkAttrs (Map.fromList pairs)
    SrcUrl UrlSrc{..} -> do
        pairs <- sequence
            [ ("type",) <$> mkString "url"
            , ("url",) <$> mkString urlUrl
            , ("hash",) <$> mkString urlHash
            ]
        mkAttrs (Map.fromList pairs)
    SrcGit GitSrc{..} -> do
        pairs <- sequence
            [ ("type",) <$> mkString "git"
            , ("url",) <$> mkString gitUrl
            , ("rev",) <$> mkString gitRev
            , ("hash",) <$> mkString gitHash
            ]
        mkAttrs (Map.fromList pairs)
    SrcStore (StorePath p) -> do
        pairs <- sequence
            [ ("type",) <$> mkString "store"
            , ("path",) <$> mkString p
            ]
        mkAttrs (Map.fromList pairs)
    SrcNone -> mkNull

-- | Convert deps to Nix
depsToNix :: [Dep] -> IO Value
depsToNix depList = do
    let byKind k = [depName d | d <- depList, depKind d == k]
    nativeBuildInputs <- mkList =<< mapM mkString (byKind Build)
    buildInputs <- mkList =<< mapM mkString (byKind Host)
    propagatedBuildInputs <- mkList =<< mapM mkString (byKind Propagate)
    checkInputs <- mkList =<< mapM mkString (byKind Check)
    mkAttrs $ Map.fromList
        [ ("nativeBuildInputs", nativeBuildInputs)
        , ("buildInputs", buildInputs)
        , ("propagatedBuildInputs", propagatedBuildInputs)
        , ("checkInputs", checkInputs)
        ]

-- | Convert Meta to Nix
metaToNix :: Meta -> IO Value
metaToNix Meta{..} = do
    homepageVal <- case homepage of
        Nothing -> mkNull
        Just h -> mkString h
    descriptionVal <- mkString description
    licenseVal <- mkString license
    platformsVal <- mkList =<< mapM mkString platforms
    mkAttrs $ Map.fromList
        [ ("description", descriptionVal)
        , ("homepage", homepageVal)
        , ("license", licenseVal)
        , ("platforms", platformsVal)
        ]

-- | Convert Phases to Nix (for legacy stdenv path)
phasesToNix :: Phases -> IO Value
phasesToNix Phases{..} = do
    pairs <- sequence
        [ ("postPatch",) <$> actionsToNix patch
        , ("preConfigure",) <$> actionsToNix configure
        , ("installPhase",) <$> actionsToNix install
        , ("postInstall",) <$> actionsToNix install  -- TODO: separate
        , ("postFixup",) <$> actionsToNix fixup
        ]
    mkAttrs (Map.fromList pairs)

-- | Convert actions to Nix (as list of action attrsets)
actionsToNix :: [Action] -> IO Value
actionsToNix actions = mkList =<< mapM actionToNix actions

-- | Convert single action to Nix attrset
actionToNix :: Action -> IO Value
actionToNix = \case
    Mkdir ref parents -> do
        pairs <- sequence
            [ ("action",) <$> mkString "mkdir"
            , ("path",) <$> mkString (refToText ref)
            ]
        mkAttrs (Map.fromList pairs)
    Write ref contents -> do
        pairs <- sequence
            [ ("action",) <$> mkString "writeFile"
            , ("path",) <$> mkString (refToText ref)
            , ("content",) <$> mkString contents
            ]
        mkAttrs (Map.fromList pairs)
    Symlink target link -> do
        pairs <- sequence
            [ ("action",) <$> mkString "symlink"
            , ("target",) <$> mkString (refToText target)
            , ("link",) <$> mkString (refToText link)
            ]
        mkAttrs (Map.fromList pairs)
    Copy src dst -> do
        pairs <- sequence
            [ ("action",) <$> mkString "copy"
            , ("src",) <$> mkString (refToText src)
            , ("dst",) <$> mkString (refToText dst)
            ]
        mkAttrs (Map.fromList pairs)
    Substitute file reps -> do
        repList <- mkList =<< mapM (\(from, to) -> do
            pairs <- sequence
                [ ("from",) <$> mkString from
                , ("to",) <$> mkString to
                ]
            mkAttrs (Map.fromList pairs)) reps
        pairs <- sequence
            [ ("action",) <$> mkString "substitute"
            , ("file",) <$> mkString (refToText file)
            , ("replacements",) <$> pure repList
            ]
        mkAttrs (Map.fromList pairs)
    PatchElfRpath path rpaths -> do
        rpathList <- mkList =<< mapM (mkString . refToText) rpaths
        pairs <- sequence
            [ ("action",) <$> mkString "patchelfRpath"
            , ("path",) <$> mkString (refToText path)
            , ("rpaths",) <$> pure rpathList
            ]
        mkAttrs (Map.fromList pairs)
    Shell cmd -> do
        pairs <- sequence
            [ ("action",) <$> mkString "run"
            , ("cmd",) <$> mkString cmd
            , ("args",) <$> mkList []
            ]
        mkAttrs (Map.fromList pairs)
    _ -> do
        -- Fallback for unhandled actions
        pairs <- sequence [("action",) <$> mkString "noop"]
        mkAttrs (Map.fromList pairs)

-- | Convert Ref to Text (for legacy path)
refToText :: Ref -> Text
refToText = \case
    RefDep name msub -> maybe name (\s -> name <> "/" <> s) msub
    RefOut name msub -> maybe ("$out") (\s -> "$out/" <> s) msub
    RefSrc msub -> maybe "$src" (\s -> "$src/" <> s) msub
    RefEnv v -> "$" <> v
    RefRel p -> p
    RefLit t -> t
    RefCat refs -> T.concat (map refToText refs)

-- | Convert env to Nix
envToNix :: [(Text, Text)] -> IO Value
envToNix envVars = do
    pairs <- mapM (\(k, v) -> (k,) <$> mkString v) envVars
    mkAttrs (Map.fromList pairs)
