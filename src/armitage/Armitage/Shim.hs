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
module Armitage.Shim
  ( -- * Metadata types
    CompileInfo (..)
  , LinkInfo (..)
  , ArchiveInfo (..)
  , BuildMetadata (..)
  
    -- * Reading metadata
  , readObjectMetadata
  , readExecutableMetadata
  , readArchiveMetadata
  , readBuildMetadata
  
    -- * Log parsing
  , ShimLogEntry (..)
  , parseShimLog
  
    -- * Shim generation
  , generateShimEnv
  , ShimPaths (..)
  ) where

import Control.Exception (try, SomeException)
import Control.Monad (forM, when)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Encoding as TE
import Data.Word (Word8, Word16, Word32, Word64)
import Data.Bits (shiftL, (.|.))
import System.Directory (doesFileExist)
import System.FilePath ((</>))

-- -----------------------------------------------------------------------------
-- Metadata Types
-- -----------------------------------------------------------------------------

-- | Metadata from a compiled object file
data CompileInfo = CompileInfo
  { ciSources   :: [Text]      -- ^ Source files compiled
  , ciIncludes  :: [Text]      -- ^ Include paths (-I, -isystem)
  , ciDefines   :: [Text]      -- ^ Preprocessor defines (-D)
  , ciFlags     :: [Text]      -- ^ Other compiler flags
  , ciOutput    :: Text        -- ^ Output file
  }
  deriving (Show, Eq)

-- | Metadata from a linked executable/library
data LinkInfo = LinkInfo
  { liObjects   :: [Text]      -- ^ Object files linked
  , liLibs      :: [Text]      -- ^ Libraries linked (-l)
  , liLibPaths  :: [Text]      -- ^ Library search paths (-L)
  , liRPaths    :: [Text]      -- ^ Runtime paths (-rpath)
  , liOutput    :: Text        -- ^ Output file
  -- Aggregated from all compiled objects:
  , liAllSources  :: [Text]
  , liAllIncludes :: [Text]
  , liAllDefines  :: [Text]
  , liAllFlags    :: [Text]
  }
  deriving (Show, Eq)

-- | Metadata from an archive
data ArchiveInfo = ArchiveInfo
  { aiArchive     :: Text        -- ^ Archive name
  , aiMembers     :: [Text]      -- ^ Member objects
  , aiAllSources  :: [Text]
  , aiAllIncludes :: [Text]
  , aiAllDefines  :: [Text]
  , aiAllFlags    :: [Text]
  }
  deriving (Show, Eq)

-- | Complete build metadata for a target
data BuildMetadata = BuildMetadata
  { bmTarget      :: Text        -- ^ Final output
  , bmSources     :: [Text]      -- ^ All source files
  , bmIncludes    :: [Text]      -- ^ All include paths  
  , bmDefines     :: [Text]      -- ^ All defines
  , bmFlags       :: [Text]      -- ^ All flags
  , bmLibs        :: [Text]      -- ^ All libraries
  , bmLibPaths    :: [Text]      -- ^ All library paths
  , bmObjects     :: [Text]      -- ^ All intermediate objects
  }
  deriving (Show, Eq)

-- -----------------------------------------------------------------------------
-- ELF Parsing (minimal, just enough to read symbols)
-- -----------------------------------------------------------------------------

-- | Read a little-endian Word16
readWord16 :: ByteString -> Int -> Word16
readWord16 bs off = 
  fromIntegral (BS.index bs off) .|.
  (fromIntegral (BS.index bs (off + 1)) `shiftL` 8)

-- | Read a little-endian Word32
readWord32 :: ByteString -> Int -> Word32
readWord32 bs off =
  fromIntegral (BS.index bs off) .|.
  (fromIntegral (BS.index bs (off + 1)) `shiftL` 8) .|.
  (fromIntegral (BS.index bs (off + 2)) `shiftL` 16) .|.
  (fromIntegral (BS.index bs (off + 3)) `shiftL` 24)

-- | Read a little-endian Word64
readWord64 :: ByteString -> Int -> Word64
readWord64 bs off =
  fromIntegral (BS.index bs off) .|.
  (fromIntegral (BS.index bs (off + 1)) `shiftL` 8) .|.
  (fromIntegral (BS.index bs (off + 2)) `shiftL` 16) .|.
  (fromIntegral (BS.index bs (off + 3)) `shiftL` 24) .|.
  (fromIntegral (BS.index bs (off + 4)) `shiftL` 32) .|.
  (fromIntegral (BS.index bs (off + 5)) `shiftL` 40) .|.
  (fromIntegral (BS.index bs (off + 6)) `shiftL` 48) .|.
  (fromIntegral (BS.index bs (off + 7)) `shiftL` 56)

-- | Read null-terminated string from ByteString
readCString :: ByteString -> Int -> Text
readCString bs off = 
  let bytes = BS.takeWhile (/= 0) (BS.drop off bs)
  in TE.decodeUtf8 bytes

-- | Extract armitage symbols from an ELF file
readElfSymbols :: ByteString -> Map Text Text
readElfSymbols bs
  | BS.length bs < 64 = Map.empty  -- Too small for ELF header
  | BS.take 4 bs /= "\x7fELF" = Map.empty  -- Not ELF
  | otherwise = 
      let shoff = fromIntegral $ readWord64 bs 40  -- e_shoff
          shentsize = fromIntegral $ readWord16 bs 58  -- e_shentsize
          shnum = fromIntegral $ readWord16 bs 60  -- e_shnum
          
          -- Find .symtab section
          findSymtab i
            | i >= shnum = Nothing
            | otherwise =
                let shdrOff = shoff + i * shentsize
                    shType = readWord32 bs shdrOff + 4  -- sh_type at offset 4
                in if shType == 2  -- SHT_SYMTAB
                   then Just (shdrOff, i)
                   else findSymtab (i + 1)
                   
      in case findSymtab 0 of
           Nothing -> Map.empty
           Just (symtabShdr, _) ->
             let symtabOff = fromIntegral $ readWord64 bs (symtabShdr + 24)  -- sh_offset
                 symtabSize = fromIntegral $ readWord64 bs (symtabShdr + 32)  -- sh_size
                 strtabIdx = fromIntegral $ readWord32 bs (symtabShdr + 40)  -- sh_link
                 
                 strtabShdr = shoff + strtabIdx * shentsize
                 strtabOff = fromIntegral $ readWord64 bs (strtabShdr + 24)
                 
                 -- Find .data section for symbol values
                 findData i
                   | i >= shnum = Nothing
                   | otherwise =
                       let shdrOff = shoff + i * shentsize
                           shType = readWord32 bs (shdrOff + 4)
                           shFlags = readWord64 bs (shdrOff + 8)
                       in if shType == 1 && shFlags == 3  -- SHT_PROGBITS, SHF_WRITE|SHF_ALLOC
                          then Just (fromIntegral $ readWord64 bs (shdrOff + 24))
                          else findData (i + 1)
                 
                 dataOff = fromMaybe 0 (findData 0)
                 
                 -- Parse symbols
                 numSyms = symtabSize `div` 24  -- sizeof(Elf64_Sym)
                 parseSyms acc i
                   | i >= numSyms = acc
                   | otherwise =
                       let symOff = symtabOff + i * 24
                           stName = fromIntegral $ readWord32 bs symOff  -- st_name
                           stValue = fromIntegral $ readWord64 bs (symOff + 8)  -- st_value
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
  if not exists then pure Nothing
  else do
    result <- try $ BS.readFile path
    case result of
      Left (_ :: SomeException) -> pure Nothing
      Right bs ->
        let syms = readElfSymbols bs
        in if Map.null syms then pure Nothing
           else pure $ Just CompileInfo
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
  if not exists then pure Nothing
  else do
    result <- try $ BS.readFile path
    case result of
      Left (_ :: SomeException) -> pure Nothing
      Right bs ->
        let syms = readElfSymbols bs
        in if Map.null syms then pure Nothing
           else pure $ Just LinkInfo
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
  if not exists then pure Nothing
  else do
    result <- try $ BS.readFile path
    case result of
      Left (_ :: SomeException) -> pure Nothing
      Right bs ->
        let syms = readElfSymbols bs
        in if Map.null syms then pure Nothing
           else pure $ Just ArchiveInfo
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
    Just li -> pure $ Just BuildMetadata
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
  , slePid       :: Int
  , sleTool      :: Text         -- ^ CC, LD, AR
  , sleArgs      :: [Text]
  }
  deriving (Show, Eq)

-- | Parse the shim log file
parseShimLog :: FilePath -> IO [ShimLogEntry]
parseShimLog path = do
  exists <- doesFileExist path
  if not exists then pure []
  else do
    content <- TIO.readFile path
    pure $ catMaybes $ map parseLogLine $ T.lines content

parseLogLine :: Text -> Maybe ShimLogEntry
parseLogLine line = do
  -- Format: [YYYY-MM-DD HH:MM:SS] pid=NNN TOOL args...
  let stripped = T.strip line
  if T.null stripped then Nothing
  else do
    -- Extract timestamp
    let (ts, rest) = T.breakOn "]" stripped
        timestamp = T.drop 1 ts  -- drop leading [
        afterTs = T.strip $ T.drop 1 rest  -- drop ]
    
    -- Extract pid
    let (pidPart, rest2) = T.breakOn " " afterTs
    pid <- if "pid=" `T.isPrefixOf` pidPart
           then readMaybe $ T.unpack $ T.drop 4 pidPart
           else Nothing
    
    let afterPid = T.strip rest2
        (tool, argsStr) = T.breakOn " " afterPid
        args = T.words argsStr
    
    Just ShimLogEntry
      { sleTimestamp = timestamp
      , slePid = pid
      , sleTool = tool
      , sleArgs = args
      }
  where
    readMaybe :: Read a => String -> Maybe a
    readMaybe s = case reads s of
      [(x, "")] -> Just x
      _ -> Nothing

-- -----------------------------------------------------------------------------
-- Shim Generation
-- -----------------------------------------------------------------------------

-- | Paths to shim binaries
data ShimPaths = ShimPaths
  { spCC    :: FilePath
  , spCXX   :: FilePath
  , spLD    :: FilePath
  , spAR    :: FilePath
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
  -- Also override via PATH-style vars that some build systems use
  , ("CMAKE_C_COMPILER", spCC)
  , ("CMAKE_CXX_COMPILER", spCXX)
  , ("CMAKE_AR", spAR)
  , ("CMAKE_LINKER", spLD)
  ]
