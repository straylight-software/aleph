{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Tar
Description : Typed wrapper for tar

This module was auto-generated from @tar --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Tar (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    tar,
    tar_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { catenate :: Bool
    -- ^ -A: , --concatenate   append tar files to an archive
    , create :: Bool
    -- ^ -c: create a new archive
    , delete :: Bool
    -- ^ delete from the archive (not on mag tapes!)
    , diff :: Bool
    -- ^ -d: , --compare      find differences between archive
    , append :: Bool
    -- ^ -r: append files to the end of an archive
    , testLabel :: Bool
    -- ^ test the archive volume label and exit
    , list :: Bool
    -- ^ -t: list the contents of an archive
    , update :: Bool
    -- ^ -u: only append files newer than copy in archive
    , extract :: Bool
    -- ^ -x: , --get       extract files from an archive
    , checkDevice :: Bool
    -- ^ check device numbers when creating incremental
    , listedIncremental :: Maybe FilePath
    -- ^ -g: handle new GNU-format incremental backup
    , incremental :: Bool
    -- ^ -G: handle old GNU-format incremental backup
    , holeDetection :: Maybe Text
    -- ^ technique to detect holes
    , ignoreFailedRead :: Bool
    -- ^ do not exit with nonzero on unreadable files
    , level :: Maybe Int
    -- ^ dump level for created listed-incremental archive
    , noCheckDevice :: Bool
    -- ^ do not check device numbers when creating
    , noSeek :: Bool
    -- ^ archive is not seekable
    , seek :: Bool
    -- ^ -n: archive is seekable
    , occurrence :: Maybe Int
    -- ^ process only the NUMBERth occurrence of each file
    , sparseVersion :: Maybe Text
    -- ^ [.MINOR]
    , sparse :: Bool
    -- ^ -S: handle sparse files efficiently
    , addFile :: Maybe FilePath
    -- ^ add given FILE to the archive (useful if its name
    , directory :: Maybe FilePath
    -- ^ -C: change to directory DIR
    , exclude :: Maybe Text
    -- ^ exclude files, given as a PATTERN
    , excludeBackups :: Bool
    -- ^ exclude backup and lock files
    , excludeCaches :: Bool
    -- ^ exclude contents of directories containing
    , excludeCachesAll :: Bool
    -- ^ exclude directories containing CACHEDIR.TAG
    , excludeCachesUnder :: Bool
    -- ^ exclude everything under directories containing
    , excludeIgnore :: Maybe FilePath
    -- ^ read exclude patterns for each directory from
    , excludeIgnoreRecursive :: Maybe FilePath
    , excludeTag :: Maybe FilePath
    -- ^ exclude contents of directories containing FILE,
    , excludeTagAll :: Maybe FilePath
    -- ^ exclude directories containing FILE
    , excludeTagUnder :: Maybe FilePath
    -- ^ exclude everything under directories
    , excludeVcs :: Bool
    -- ^ exclude version control system directories
    , excludeVcsIgnores :: Bool
    -- ^ read exclude patterns from the VCS ignore files
    , noNull :: Bool
    -- ^ disable the effect of the previous --null option
    , noRecursion :: Bool
    -- ^ avoid descending automatically in directories
    , noUnquote :: Bool
    -- ^ do not unquote input file or member names
    , noVerbatimFilesFrom :: Maybe Text
    -- ^ treats file names starting with dash as
    , null_ :: Maybe Text
    -- ^ reads null-terminated names; implies
    , recursion :: Bool
    -- ^ recurse into directories (default)
    , filesFrom :: Maybe FilePath
    -- ^ -T: get names to extract or create from FILE
    , unquote :: Bool
    -- ^ unquote input file or member names (default)
    , verbatimFilesFrom :: Maybe Text
    -- ^ reads file names verbatim (no escape or option
    , excludeFrom :: Maybe FilePath
    -- ^ -X: exclude patterns listed in FILE
    , anchored :: Bool
    -- ^ patterns match file name start
    , ignoreCase :: Bool
    -- ^ ignore case
    , noAnchored :: Bool
    -- ^ patterns match after any '/' (default for
    , noIgnoreCase :: Bool
    -- ^ case sensitive matching (default)
    , noWildcards :: Bool
    -- ^ verbatim string matching
    , noWildcardsMatchSlash :: Bool
    -- ^ wildcards do not match '/'
    , wildcards :: Bool
    -- ^ use wildcards (default for exclusion)
    , wildcardsMatchSlash :: Bool
    -- ^ wildcards match '/' (default for exclusion)
    , keepDirectorySymlink :: Bool
    -- ^ preserve existing symlinks to directories when
    , keepNewerFiles :: Bool
    -- ^ don't replace existing files that are newer than
    , keepOldFiles :: Bool
    -- ^ -k: don't replace existing files when extracting,
    , noOverwriteDir :: Bool
    -- ^ preserve metadata of existing directories
    , oneTopLevel :: Maybe FilePath
    -- ^ create a subdirectory to avoid having loose files
    , overwrite :: Bool
    -- ^ overwrite existing files when extracting
    , overwriteDir :: Bool
    -- ^ overwrite metadata of existing directories when
    , recursiveUnlink :: Bool
    -- ^ empty hierarchies prior to extracting directory
    , removeFiles :: Bool
    -- ^ remove files after adding them to the archive
    , skipOldFiles :: Bool
    -- ^ don't replace existing files when extracting,
    , unlinkFirst :: Bool
    -- ^ -U: remove each file prior to extracting over it
    , verify :: Bool
    -- ^ -W: attempt to verify the archive after writing it
    , ignoreCommandError :: Bool
    -- ^ ignore exit codes of children
    , noIgnoreCommandError :: Bool
    -- ^ treat non-zero exit codes of children as
    , toStdout :: Bool
    -- ^ -O: extract files to standard output
    , toCommand :: Maybe Text
    -- ^ pipe extracted files to another program
    , atimePreserve :: Maybe Text
    -- ^ preserve access times on dumped files, either
    , clampMtime :: Bool
    -- ^ only set time when the file is more recent than
    , delayDirectoryRestore :: Bool
    -- ^ delay setting modification times and
    , group :: Maybe Text
    -- ^ force NAME as group for added files
    , groupMap :: Maybe FilePath
    -- ^ use FILE to map file owner GIDs and names
    , mode :: Maybe Text
    -- ^ force (symbolic) mode CHANGES for added files
    , mtime :: Maybe Text
    -- ^ set mtime for added files from DATE-OR-FILE
    , touch :: Bool
    -- ^ -m: don't extract file modified time
    , noDelayDirectoryRestore :: Bool
    , noSameOwner :: Bool
    -- ^ extract files as yourself (default for ordinary
    , noSamePermissions :: Bool
    -- ^ apply the user's umask when extracting permissions
    , numericOwner :: Bool
    -- ^ always use numbers for user/group names
    , owner :: Maybe Text
    -- ^ force NAME as owner for added files
    , ownerMap :: Maybe FilePath
    -- ^ use FILE to map file owner UIDs and names
    , preservePermissions :: Bool
    -- ^ -p: , --same-permissions
    , sameOwner :: Bool
    -- ^ try extracting files with the same ownership as
    , sort :: Maybe Text
    -- ^ directory sorting order: none (default), name or
    , preserveOrder :: Bool
    -- ^ -s: , --same-order
    , acls :: Maybe Text
    -- ^ nable the POSIX ACLs support
    , noAcls :: Maybe Text
    -- ^ isable the POSIX ACLs support
    , noSelinux :: Maybe Text
    -- ^ isable the SELinux context support
    , noXattrs :: Maybe Text
    -- ^ isable extended attributes support
    , selinux :: Maybe Text
    -- ^ nable the SELinux context support
    , xattrs :: Maybe Text
    -- ^ nable extended attributes support
    , xattrsExclude :: Maybe Text
    -- ^ specify the exclude pattern for xattr keys
    , xattrsInclude :: Maybe Text
    -- ^ specify the include pattern for xattr keys
    , forceLocal :: Bool
    -- ^ archive file is local even if it has a colon
    , file :: Maybe Text
    -- ^ -f: use archive file or device ARCHIVE
    , infoScript :: Maybe Text
    -- ^ -F: , --new-volume-script=NAME
    , tapeLength :: Maybe Int
    -- ^ -L: change tape after writing NUMBER x 1024 bytes
    , multiVolume :: Bool
    -- ^ -M: create/list/extract multi-volume archive
    , rmtCommand :: Maybe Text
    -- ^ use given rmt COMMAND instead of rmt
    , rshCommand :: Maybe Text
    -- ^ use remote COMMAND instead of rsh
    , volnoFile :: Maybe FilePath
    -- ^ use/update the volume number in FILE
    , blockingFactor :: Maybe Text
    -- ^ -b: BLOCKS x 512 bytes per record
    , readFullRecords :: Bool
    -- ^ -B: reblock as we read (for 4.2BSD pipes)
    , ignoreZeros :: Bool
    -- ^ -i: ignore zeroed blocks in archive (means EOF)
    , recordSize :: Maybe Int
    -- ^ NUMBER of bytes per record, multiple of 512
    , format :: Maybe Text
    -- ^ -H: create archive of the given format
    , oldArchive :: Bool
    -- ^ , --portability
    , paxOption :: Bool
    -- ^ =keyword[[:]=value][,keyword[[:]=value]]...
    , posix :: Bool
    -- ^ same as --format=posix
    , label :: Maybe Text
    -- ^ -V: create archive with volume name TEXT; at
    , autoCompress :: Bool
    -- ^ -a: use archive suffix to determine the compression
    , useCompressProgram :: Maybe Text
    , bzip2 :: Bool
    -- ^ -j: filter the archive through bzip2
    , xz :: Bool
    -- ^ -J: filter the archive through xz
    , lzip :: Bool
    -- ^ filter the archive through lzip
    , lzma :: Bool
    -- ^ filter the archive through lzma
    , lzop :: Bool
    -- ^ filter the archive through lzop
    , noAutoCompress :: Bool
    -- ^ do not use archive suffix to determine the
    , zstd :: Bool
    -- ^ filter the archive through zstd
    , gzip :: Bool
    -- ^ -z: , --gunzip, --ungzip   filter the archive through
    , compress :: Bool
    -- ^ -Z: , --uncompress   filter the archive through compre
    , backup :: Maybe Text
    -- ^ backup before removal, choose version CONTROL
    , hardDereference :: Bool
    -- ^ follow hard links; archive and dump the files they
    , dereference :: Bool
    -- ^ -h: follow symlinks; archive and dump the files they
    , startingFile :: Maybe Text
    , newerMtime :: Maybe Text
    -- ^ compare date and time when data changed only
    , newer :: Maybe Text
    -- ^ -N: , --after-date=DATE-OR-FILE
    , oneFileSystem :: Bool
    -- ^ stay in local file system when creating archive
    , absoluteNames :: Bool
    -- ^ -P: don't strip leading '/'s from file names
    , suffix :: Maybe Text
    -- ^ backup before removal, override usual suffix ('~'
    , stripComponents :: Maybe Int
    -- ^ strip NUMBER leading components from file
    , transform :: Maybe Text
    -- ^ , --xform=EXPRESSION
    , checkpoint :: Maybe Int
    -- ^ display progress messages every NUMBERth record
    , checkpointAction :: Maybe Text
    -- ^ execute ACTION on each checkpoint
    , fullTime :: Bool
    -- ^ print file time to its full resolution
    , indexFile :: Maybe FilePath
    -- ^ send verbose output to FILE
    , checkLinks :: Bool
    -- ^ -l: print a message if not all links are dumped
    , noQuoteChars :: Maybe Text
    -- ^ disable quoting for characters from STRING
    , quoteChars :: Maybe Text
    -- ^ additionally quote characters from STRING
    , quotingStyle :: Maybe Text
    -- ^ set name quoting style; see below for valid STYLE
    , blockNumber :: Bool
    -- ^ -R: show block number within archive with each message
    , showDefaults :: Bool
    -- ^ show tar defaults
    , showOmittedDirs :: Bool
    -- ^ when listing or extracting, list each directory
    , showSnapshotFieldRanges :: Bool
    , showTransformedNames :: Bool
    -- ^ , --show-stored-names
    , totals :: Maybe Text
    -- ^ print total bytes after processing the archive;
    , utc :: Bool
    -- ^ print file modification times in UTC
    , verbose :: Bool
    -- ^ -v: verbosely list files processed
    , warning :: Maybe Text
    -- ^ warning control
    , interactive :: Bool
    -- ^ -w: , --confirmation
    , optO :: Bool
    -- ^ -o: when creating, same as --old-archive; when
    , restrict :: Bool
    -- ^ disable use of some potentially harmful options
    , usage :: Bool
    -- ^ give a short usage message
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { catenate = False
        , create = False
        , delete = False
        , diff = False
        , append = False
        , testLabel = False
        , list = False
        , update = False
        , extract = False
        , checkDevice = False
        , listedIncremental = Nothing
        , incremental = False
        , holeDetection = Nothing
        , ignoreFailedRead = False
        , level = Nothing
        , noCheckDevice = False
        , noSeek = False
        , seek = False
        , occurrence = Nothing
        , sparseVersion = Nothing
        , sparse = False
        , addFile = Nothing
        , directory = Nothing
        , exclude = Nothing
        , excludeBackups = False
        , excludeCaches = False
        , excludeCachesAll = False
        , excludeCachesUnder = False
        , excludeIgnore = Nothing
        , excludeIgnoreRecursive = Nothing
        , excludeTag = Nothing
        , excludeTagAll = Nothing
        , excludeTagUnder = Nothing
        , excludeVcs = False
        , excludeVcsIgnores = False
        , noNull = False
        , noRecursion = False
        , noUnquote = False
        , noVerbatimFilesFrom = Nothing
        , null_ = Nothing
        , recursion = False
        , filesFrom = Nothing
        , unquote = False
        , verbatimFilesFrom = Nothing
        , excludeFrom = Nothing
        , anchored = False
        , ignoreCase = False
        , noAnchored = False
        , noIgnoreCase = False
        , noWildcards = False
        , noWildcardsMatchSlash = False
        , wildcards = False
        , wildcardsMatchSlash = False
        , keepDirectorySymlink = False
        , keepNewerFiles = False
        , keepOldFiles = False
        , noOverwriteDir = False
        , oneTopLevel = Nothing
        , overwrite = False
        , overwriteDir = False
        , recursiveUnlink = False
        , removeFiles = False
        , skipOldFiles = False
        , unlinkFirst = False
        , verify = False
        , ignoreCommandError = False
        , noIgnoreCommandError = False
        , toStdout = False
        , toCommand = Nothing
        , atimePreserve = Nothing
        , clampMtime = False
        , delayDirectoryRestore = False
        , group = Nothing
        , groupMap = Nothing
        , mode = Nothing
        , mtime = Nothing
        , touch = False
        , noDelayDirectoryRestore = False
        , noSameOwner = False
        , noSamePermissions = False
        , numericOwner = False
        , owner = Nothing
        , ownerMap = Nothing
        , preservePermissions = False
        , sameOwner = False
        , sort = Nothing
        , preserveOrder = False
        , acls = Nothing
        , noAcls = Nothing
        , noSelinux = Nothing
        , noXattrs = Nothing
        , selinux = Nothing
        , xattrs = Nothing
        , xattrsExclude = Nothing
        , xattrsInclude = Nothing
        , forceLocal = False
        , file = Nothing
        , infoScript = Nothing
        , tapeLength = Nothing
        , multiVolume = False
        , rmtCommand = Nothing
        , rshCommand = Nothing
        , volnoFile = Nothing
        , blockingFactor = Nothing
        , readFullRecords = False
        , ignoreZeros = False
        , recordSize = Nothing
        , format = Nothing
        , oldArchive = False
        , paxOption = False
        , posix = False
        , label = Nothing
        , autoCompress = False
        , useCompressProgram = Nothing
        , bzip2 = False
        , xz = False
        , lzip = False
        , lzma = False
        , lzop = False
        , noAutoCompress = False
        , zstd = False
        , gzip = False
        , compress = False
        , backup = Nothing
        , hardDereference = False
        , dereference = False
        , startingFile = Nothing
        , newerMtime = Nothing
        , newer = Nothing
        , oneFileSystem = False
        , absoluteNames = False
        , suffix = Nothing
        , stripComponents = Nothing
        , transform = Nothing
        , checkpoint = Nothing
        , checkpointAction = Nothing
        , fullTime = False
        , indexFile = Nothing
        , checkLinks = False
        , noQuoteChars = Nothing
        , quoteChars = Nothing
        , quotingStyle = Nothing
        , blockNumber = False
        , showDefaults = False
        , showOmittedDirs = False
        , showSnapshotFieldRanges = False
        , showTransformedNames = False
        , totals = Nothing
        , utc = False
        , verbose = False
        , warning = Nothing
        , interactive = False
        , optO = False
        , restrict = False
        , usage = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag catenate "--catenate"
        , flag create "--create"
        , flag delete "--delete"
        , flag diff "--diff"
        , flag append "--append"
        , flag testLabel "--test-label"
        , flag list "--list"
        , flag update "--update"
        , flag extract "--extract"
        , flag checkDevice "--check-device"
        , optShow listedIncremental "--listed-incremental"
        , flag incremental "--incremental"
        , opt holeDetection "--hole-detection"
        , flag ignoreFailedRead "--ignore-failed-read"
        , optShow level "--level"
        , flag noCheckDevice "--no-check-device"
        , flag noSeek "--no-seek"
        , flag seek "--seek"
        , optShow occurrence "--occurrence"
        , opt sparseVersion "--sparse-version"
        , flag sparse "--sparse"
        , optShow addFile "--add-file"
        , optShow directory "--directory"
        , opt exclude "--exclude"
        , flag excludeBackups "--exclude-backups"
        , flag excludeCaches "--exclude-caches"
        , flag excludeCachesAll "--exclude-caches-all"
        , flag excludeCachesUnder "--exclude-caches-under"
        , optShow excludeIgnore "--exclude-ignore"
        , optShow excludeIgnoreRecursive "--exclude-ignore-recursive"
        , optShow excludeTag "--exclude-tag"
        , optShow excludeTagAll "--exclude-tag-all"
        , optShow excludeTagUnder "--exclude-tag-under"
        , flag excludeVcs "--exclude-vcs"
        , flag excludeVcsIgnores "--exclude-vcs-ignores"
        , flag noNull "--no-null"
        , flag noRecursion "--no-recursion"
        , flag noUnquote "--no-unquote"
        , opt noVerbatimFilesFrom "--no-verbatim-files-from"
        , opt null_ "--null"
        , flag recursion "--recursion"
        , optShow filesFrom "--files-from"
        , flag unquote "--unquote"
        , opt verbatimFilesFrom "--verbatim-files-from"
        , optShow excludeFrom "--exclude-from"
        , flag anchored "--anchored"
        , flag ignoreCase "--ignore-case"
        , flag noAnchored "--no-anchored"
        , flag noIgnoreCase "--no-ignore-case"
        , flag noWildcards "--no-wildcards"
        , flag noWildcardsMatchSlash "--no-wildcards-match-slash"
        , flag wildcards "--wildcards"
        , flag wildcardsMatchSlash "--wildcards-match-slash"
        , flag keepDirectorySymlink "--keep-directory-symlink"
        , flag keepNewerFiles "--keep-newer-files"
        , flag keepOldFiles "--keep-old-files"
        , flag noOverwriteDir "--no-overwrite-dir"
        , optShow oneTopLevel "--one-top-level"
        , flag overwrite "--overwrite"
        , flag overwriteDir "--overwrite-dir"
        , flag recursiveUnlink "--recursive-unlink"
        , flag removeFiles "--remove-files"
        , flag skipOldFiles "--skip-old-files"
        , flag unlinkFirst "--unlink-first"
        , flag verify "--verify"
        , flag ignoreCommandError "--ignore-command-error"
        , flag noIgnoreCommandError "--no-ignore-command-error"
        , flag toStdout "--to-stdout"
        , opt toCommand "--to-command"
        , opt atimePreserve "--atime-preserve"
        , flag clampMtime "--clamp-mtime"
        , flag delayDirectoryRestore "--delay-directory-restore"
        , opt group "--group"
        , optShow groupMap "--group-map"
        , opt mode "--mode"
        , opt mtime "--mtime"
        , flag touch "--touch"
        , flag noDelayDirectoryRestore "--no-delay-directory-restore"
        , flag noSameOwner "--no-same-owner"
        , flag noSamePermissions "--no-same-permissions"
        , flag numericOwner "--numeric-owner"
        , opt owner "--owner"
        , optShow ownerMap "--owner-map"
        , flag preservePermissions "--preserve-permissions"
        , flag sameOwner "--same-owner"
        , opt sort "--sort"
        , flag preserveOrder "--preserve-order"
        , opt acls "--acls"
        , opt noAcls "--no-acls"
        , opt noSelinux "--no-selinux"
        , opt noXattrs "--no-xattrs"
        , opt selinux "--selinux"
        , opt xattrs "--xattrs"
        , opt xattrsExclude "--xattrs-exclude"
        , opt xattrsInclude "--xattrs-include"
        , flag forceLocal "--force-local"
        , opt file "--file"
        , opt infoScript "--info-script"
        , optShow tapeLength "--tape-length"
        , flag multiVolume "--multi-volume"
        , opt rmtCommand "--rmt-command"
        , opt rshCommand "--rsh-command"
        , optShow volnoFile "--volno-file"
        , opt blockingFactor "--blocking-factor"
        , flag readFullRecords "--read-full-records"
        , flag ignoreZeros "--ignore-zeros"
        , optShow recordSize "--record-size"
        , opt format "--format"
        , flag oldArchive "--old-archive"
        , flag paxOption "--pax-option"
        , flag posix "--posix"
        , opt label "--label"
        , flag autoCompress "--auto-compress"
        , opt useCompressProgram "--use-compress-program"
        , flag bzip2 "--bzip2"
        , flag xz "--xz"
        , flag lzip "--lzip"
        , flag lzma "--lzma"
        , flag lzop "--lzop"
        , flag noAutoCompress "--no-auto-compress"
        , flag zstd "--zstd"
        , flag gzip "--gzip"
        , flag compress "--compress"
        , opt backup "--backup"
        , flag hardDereference "--hard-dereference"
        , flag dereference "--dereference"
        , opt startingFile "--starting-file"
        , opt newerMtime "--newer-mtime"
        , opt newer "--newer"
        , flag oneFileSystem "--one-file-system"
        , flag absoluteNames "--absolute-names"
        , opt suffix "--suffix"
        , optShow stripComponents "--strip-components"
        , opt transform "--transform"
        , optShow checkpoint "--checkpoint"
        , opt checkpointAction "--checkpoint-action"
        , flag fullTime "--full-time"
        , optShow indexFile "--index-file"
        , flag checkLinks "--check-links"
        , opt noQuoteChars "--no-quote-chars"
        , opt quoteChars "--quote-chars"
        , opt quotingStyle "--quoting-style"
        , flag blockNumber "--block-number"
        , flag showDefaults "--show-defaults"
        , flag showOmittedDirs "--show-omitted-dirs"
        , flag showSnapshotFieldRanges "--show-snapshot-field-ranges"
        , flag showTransformedNames "--show-transformed-names"
        , opt totals "--totals"
        , flag utc "--utc"
        , flag verbose "--verbose"
        , opt warning "--warning"
        , flag interactive "--interactive"
        , flag optO "-o"
        , flag restrict "--restrict"
        , flag usage "--usage"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run tar with options and additional arguments
tar :: Options -> [Text] -> Sh Text
tar opts args = run "tar" (buildArgs opts ++ args)

-- | Run tar, ignoring output
tar_ :: Options -> [Text] -> Sh ()
tar_ opts args = run_ "tar" (buildArgs opts ++ args)
