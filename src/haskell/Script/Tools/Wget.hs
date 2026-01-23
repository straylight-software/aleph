{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Wget
Description : Typed wrapper for wget

This module was auto-generated from @wget --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Wget (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    wget,
    wget_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { optV :: Maybe Text
    -- ^ -V: version                   display the version of W
    , optH :: Maybe Text
    -- ^ -h: help                      print this help
    , optB :: Maybe Text
    -- ^ -b: background                go to background after s
    , optE :: Maybe Text
    -- ^ -e: execute=COMMAND           execute a `.wgetrc'-styl
    , optO :: Maybe Text
    -- ^ -o: output-file=FILE          log messages to FILE
    , optA :: Maybe Text
    -- ^ -a: append-output=FILE        append messages to FILE
    , optD :: Maybe Text
    -- ^ -d: debug                     print lots of debugging
    , optQ :: Maybe Text
    -- ^ -q: quiet                     quiet (no output)
    , optN :: Bool
    -- ^ -n: v, --no-verbose                turn off verbosenes
    , reportSpeed :: Maybe Text
    -- ^ output bandwidth as TYPE.  TYPE can be bits
    , optI :: Maybe Text
    -- ^ -i: input-file=FILE           download URLs found in l
    , optF :: Maybe Text
    -- ^ -F: force-html                treat input file as HTML
    , config :: Maybe FilePath
    -- ^ specify config file to use
    , noConfig :: Bool
    -- ^ do not read any config file
    , rejectedLog :: Maybe FilePath
    -- ^ log reasons for URL rejection to FILE
    , optT :: Maybe Text
    -- ^ -t: tries=NUMBER              set number of retries to
    , retryConnrefused :: Bool
    -- ^ retry even if connection is refused
    , retryOnHostError :: Bool
    -- ^ consider host errors as non-fatal, transient error
    , retryOnHttpError :: Maybe Text
    -- ^ comma-separated list of HTTP errors to retry
    , noNetrc :: Bool
    -- ^ don't try to obtain credentials from .netrc
    , optC :: Maybe Text
    -- ^ -c: continue                  resume getting a partial
    , startPos :: Maybe Text
    -- ^ start downloading from zero-based position OFFSET
    , progress :: Maybe Text
    -- ^ select progress gauge type
    , showProgress :: Bool
    -- ^ display the progress bar in any verbosity mode
    , noIfModifiedSince :: Bool
    -- ^ don't use conditional if-modified-since get
    , noUseServerTimestamps :: Bool
    -- ^ don't set the local file's timestamp by
    , optS :: Maybe Text
    -- ^ -S: server-response           print server response
    , spider :: Bool
    -- ^ don't download anything
    , dnsTimeout :: Maybe Int
    -- ^ set the DNS lookup timeout to SECS
    , connectTimeout :: Maybe Int
    -- ^ set the connect timeout to SECS
    , readTimeout :: Maybe Int
    -- ^ set the read timeout to SECS
    , optW :: Maybe Text
    -- ^ -w: wait=SECONDS              wait SECONDS between ret
    , waitretry :: Maybe Int
    -- ^ wait 1..SECONDS between retries of a retrieval
    , randomWait :: Bool
    -- ^ wait from 0.5*WAIT...1.5*WAIT secs between retriev
    , noProxy :: Bool
    -- ^ explicitly turn off proxy
    , bindAddress :: Maybe Text
    -- ^ bind to ADDRESS (hostname or IP) on local host
    , limitRate :: Maybe Text
    -- ^ limit download rate to RATE
    , noDnsCache :: Bool
    -- ^ disable caching DNS lookups
    , restrictFileNames :: Maybe Text
    -- ^ restrict chars in file names to ones OS allows
    , ignoreCase :: Bool
    -- ^ ignore case when matching files/directories
    , opt4 :: Maybe Text
    -- ^ -4: inet4-only                connect only to IPv4 add
    , opt6 :: Maybe Text
    -- ^ -6: inet6-only                connect only to IPv6 add
    , preferFamily :: Maybe Text
    -- ^ connect first to addresses of specified family,
    , user :: Maybe Text
    -- ^ set both ftp and http user to USER
    , password :: Maybe Text
    -- ^ set both ftp and http password to PASS
    , askPassword :: Bool
    -- ^ prompt for passwords
    , useAskpass :: Maybe Text
    -- ^ specify credential handler for requesting
    , noIri :: Bool
    -- ^ turn off IRI support
    , localEncoding :: Maybe Text
    -- ^ use ENC as the local encoding for IRIs
    , remoteEncoding :: Maybe Text
    -- ^ use ENC as the default remote encoding
    , unlink :: Bool
    -- ^ remove file before clobber
    , xattr :: Bool
    -- ^ turn on storage of metadata in extended file attri
    , optX :: Maybe Text
    -- ^ -x: force-directories         force creation of direct
    , protocolDirectories :: Bool
    -- ^ use protocol name in directories
    , optP :: Maybe Text
    -- ^ -P: directory-prefix=PREFIX   save files to PREFIX/..
    , cutDirs :: Maybe Int
    -- ^ ignore NUMBER remote directory components
    , httpUser :: Maybe Text
    -- ^ set http user to USER
    , httpPassword :: Maybe Text
    -- ^ set http password to PASS
    , noCache :: Bool
    -- ^ disallow server-cached data
    , defaultPage :: Maybe Text
    -- ^ change the default page name (normally
    , ignoreLength :: Bool
    -- ^ ignore 'Content-Length' header field
    , header :: Maybe Text
    -- ^ insert STRING among the headers
    , compression :: Maybe Text
    -- ^ choose compression, one of auto, gzip and none. (d
    , maxRedirect :: Bool
    -- ^ maximum redirections allowed per page
    , proxyUser :: Maybe Text
    -- ^ set USER as proxy username
    , proxyPassword :: Maybe Text
    -- ^ set PASS as proxy password
    , referer :: Maybe Text
    -- ^ include 'Referer: URL' header in HTTP request
    , saveHeaders :: Bool
    -- ^ save the HTTP headers to file
    , optU :: Maybe Text
    -- ^ -U: user-agent=AGENT          identify as AGENT instea
    , noHttpKeepAlive :: Bool
    -- ^ disable HTTP keep-alive (persistent connections)
    , noCookies :: Bool
    -- ^ don't use cookies
    , loadCookies :: Maybe FilePath
    -- ^ load cookies from FILE before session
    , saveCookies :: Maybe FilePath
    -- ^ save cookies to FILE after session
    , keepSessionCookies :: Bool
    -- ^ load and save session (non-permanent) cookies
    , postData :: Maybe Text
    -- ^ use the POST method; send STRING as the data
    , postFile :: Maybe FilePath
    -- ^ use the POST method; send contents of FILE
    , method :: Maybe Text
    -- ^ ethod         use method "HTTPMethod" in the reque
    , bodyData :: Maybe Text
    -- ^ send STRING as data. --method MUST be set
    , bodyFile :: Maybe FilePath
    -- ^ send contents of FILE. --method MUST be set
    , contentDisposition :: Bool
    -- ^ honor the Content-Disposition header when
    , contentOnError :: Bool
    -- ^ output the received content on server errors
    , authNoChallenge :: Bool
    -- ^ send Basic HTTP authentication information
    , secureProtocol :: Maybe Text
    -- ^ choose secure protocol, one of auto, SSLv2,
    , httpsOnly :: Bool
    -- ^ only follow secure HTTPS links
    , noCheckCertificate :: Bool
    -- ^ don't validate the server's certificate
    , certificate :: Maybe FilePath
    -- ^ client certificate file
    , certificateType :: Maybe Text
    -- ^ client certificate type, PEM or DER
    , privateKey :: Maybe FilePath
    -- ^ private key file
    , privateKeyType :: Maybe Text
    -- ^ private key type, PEM or DER
    , caCertificate :: Maybe FilePath
    -- ^ file with the bundle of CAs
    , caDirectory :: Maybe FilePath
    -- ^ directory where hash list of CAs is stored
    , crlFile :: Maybe FilePath
    -- ^ file with bundle of CRLs
    , pinnedpubkey :: Maybe FilePath
    -- ^ /HASHES  Public key (PEM/DER) file, or any number
    , randomFile :: Maybe FilePath
    -- ^ file with random data for seeding the SSL PRNG
    , ciphers :: Maybe Text
    -- ^ Set the priority string (GnuTLS) or cipher list st
    , noHsts :: Bool
    -- ^ disable HSTS
    , hstsFile :: Bool
    -- ^ path of HSTS database (will override default)
    , ftpUser :: Maybe Text
    -- ^ set ftp user to USER
    , ftpPassword :: Maybe Text
    -- ^ set ftp password to PASS
    , noRemoveListing :: Bool
    -- ^ don't remove '.listing' files
    , noGlob :: Bool
    -- ^ turn off FTP file name globbing
    , noPassiveFtp :: Bool
    -- ^ disable the "passive" transfer mode
    , preservePermissions :: Bool
    -- ^ preserve remote file permissions
    , retrSymlinks :: Bool
    -- ^ when recursing, get linked-to files (not dir)
    , ftpsImplicit :: Bool
    -- ^ use implicit FTPS (default port is 990)
    , ftpsResumeSsl :: Bool
    -- ^ resume the SSL/TLS session started in the control
    , ftpsClearDataConnection :: Bool
    -- ^ cipher the control channel only; all the data will
    , ftpsFallbackToFtp :: Bool
    -- ^ fall back to FTP if FTPS is not supported in the t
    , warcFile :: Maybe Text
    -- ^ save request/response data to a .warc.gz file
    , warcHeader :: Maybe Text
    -- ^ insert STRING into the warcinfo record
    , warcMaxSize :: Maybe Int
    -- ^ set maximum size of WARC files to NUMBER
    , warcCdx :: Bool
    -- ^ write CDX index files
    , warcDedup :: Maybe Text
    -- ^ do not store records listed in this CDX file
    , noWarcCompression :: Bool
    -- ^ do not compress WARC files with GZIP
    , noWarcDigests :: Bool
    -- ^ do not calculate SHA1 digests
    , noWarcKeepLog :: Bool
    -- ^ do not store the log file in a WARC record
    , warcTempdir :: Maybe FilePath
    -- ^ location for temporary files created by the
    , optR :: Maybe Text
    -- ^ -r: recursive                 specify recursive downlo
    , optL :: Maybe Text
    -- ^ -l: level=NUMBER              maximum recursion depth
    , deleteAfter :: Bool
    -- ^ delete files locally after downloading them
    , optK :: Maybe Text
    -- ^ -k: convert-links             make links in downloaded
    , convertFileOnly :: Bool
    -- ^ convert the file part of the URLs only (usually kn
    , backups :: Maybe Int
    -- ^ before writing file X, rotate up to N backup files
    , optM :: Maybe Text
    -- ^ -m: mirror                    shortcut for -N -r -l in
    , strictComments :: Bool
    -- ^ turn on strict (SGML) handling of HTML comments
    , acceptRegex :: Maybe Text
    -- ^ regex matching accepted URLs
    , rejectRegex :: Maybe Text
    -- ^ regex matching rejected URLs
    , regexType :: Maybe Text
    -- ^ regex type (posix|pcre)
    , excludeDomains :: Maybe Text
    -- ^ comma-separated list of rejected domains
    , followFtp :: Bool
    -- ^ follow FTP links from HTML documents
    , followTags :: Maybe Text
    -- ^ comma-separated list of followed HTML tags
    , ignoreTags :: Maybe Text
    -- ^ comma-separated list of ignored HTML tags
    , trustServerNames :: Bool
    -- ^ use the name specified by the redirection
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { optV = Nothing
        , optH = Nothing
        , optB = Nothing
        , optE = Nothing
        , optO = Nothing
        , optA = Nothing
        , optD = Nothing
        , optQ = Nothing
        , optN = False
        , reportSpeed = Nothing
        , optI = Nothing
        , optF = Nothing
        , config = Nothing
        , noConfig = False
        , rejectedLog = Nothing
        , optT = Nothing
        , retryConnrefused = False
        , retryOnHostError = False
        , retryOnHttpError = Nothing
        , noNetrc = False
        , optC = Nothing
        , startPos = Nothing
        , progress = Nothing
        , showProgress = False
        , noIfModifiedSince = False
        , noUseServerTimestamps = False
        , optS = Nothing
        , spider = False
        , dnsTimeout = Nothing
        , connectTimeout = Nothing
        , readTimeout = Nothing
        , optW = Nothing
        , waitretry = Nothing
        , randomWait = False
        , noProxy = False
        , bindAddress = Nothing
        , limitRate = Nothing
        , noDnsCache = False
        , restrictFileNames = Nothing
        , ignoreCase = False
        , opt4 = Nothing
        , opt6 = Nothing
        , preferFamily = Nothing
        , user = Nothing
        , password = Nothing
        , askPassword = False
        , useAskpass = Nothing
        , noIri = False
        , localEncoding = Nothing
        , remoteEncoding = Nothing
        , unlink = False
        , xattr = False
        , optX = Nothing
        , protocolDirectories = False
        , optP = Nothing
        , cutDirs = Nothing
        , httpUser = Nothing
        , httpPassword = Nothing
        , noCache = False
        , defaultPage = Nothing
        , ignoreLength = False
        , header = Nothing
        , compression = Nothing
        , maxRedirect = False
        , proxyUser = Nothing
        , proxyPassword = Nothing
        , referer = Nothing
        , saveHeaders = False
        , optU = Nothing
        , noHttpKeepAlive = False
        , noCookies = False
        , loadCookies = Nothing
        , saveCookies = Nothing
        , keepSessionCookies = False
        , postData = Nothing
        , postFile = Nothing
        , method = Nothing
        , bodyData = Nothing
        , bodyFile = Nothing
        , contentDisposition = False
        , contentOnError = False
        , authNoChallenge = False
        , secureProtocol = Nothing
        , httpsOnly = False
        , noCheckCertificate = False
        , certificate = Nothing
        , certificateType = Nothing
        , privateKey = Nothing
        , privateKeyType = Nothing
        , caCertificate = Nothing
        , caDirectory = Nothing
        , crlFile = Nothing
        , pinnedpubkey = Nothing
        , randomFile = Nothing
        , ciphers = Nothing
        , noHsts = False
        , hstsFile = False
        , ftpUser = Nothing
        , ftpPassword = Nothing
        , noRemoveListing = False
        , noGlob = False
        , noPassiveFtp = False
        , preservePermissions = False
        , retrSymlinks = False
        , ftpsImplicit = False
        , ftpsResumeSsl = False
        , ftpsClearDataConnection = False
        , ftpsFallbackToFtp = False
        , warcFile = Nothing
        , warcHeader = Nothing
        , warcMaxSize = Nothing
        , warcCdx = False
        , warcDedup = Nothing
        , noWarcCompression = False
        , noWarcDigests = False
        , noWarcKeepLog = False
        , warcTempdir = Nothing
        , optR = Nothing
        , optL = Nothing
        , deleteAfter = False
        , optK = Nothing
        , convertFileOnly = False
        , backups = Nothing
        , optM = Nothing
        , strictComments = False
        , acceptRegex = Nothing
        , rejectRegex = Nothing
        , regexType = Nothing
        , excludeDomains = Nothing
        , followFtp = False
        , followTags = Nothing
        , ignoreTags = Nothing
        , trustServerNames = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt optV "-V"
        , opt optH "-h"
        , opt optB "-b"
        , opt optE "-e"
        , opt optO "-o"
        , opt optA "-a"
        , opt optD "-d"
        , opt optQ "-q"
        , flag optN "-n"
        , opt reportSpeed "--report-speed"
        , opt optI "-i"
        , opt optF "-F"
        , optShow config "--config"
        , flag noConfig "--no-config"
        , optShow rejectedLog "--rejected-log"
        , opt optT "-t"
        , flag retryConnrefused "--retry-connrefused"
        , flag retryOnHostError "--retry-on-host-error"
        , opt retryOnHttpError "--retry-on-http-error"
        , flag noNetrc "--no-netrc"
        , opt optC "-c"
        , opt startPos "--start-pos"
        , opt progress "--progress"
        , flag showProgress "--show-progress"
        , flag noIfModifiedSince "--no-if-modified-since"
        , flag noUseServerTimestamps "--no-use-server-timestamps"
        , opt optS "-S"
        , flag spider "--spider"
        , optShow dnsTimeout "--dns-timeout"
        , optShow connectTimeout "--connect-timeout"
        , optShow readTimeout "--read-timeout"
        , opt optW "-w"
        , optShow waitretry "--waitretry"
        , flag randomWait "--random-wait"
        , flag noProxy "--no-proxy"
        , opt bindAddress "--bind-address"
        , opt limitRate "--limit-rate"
        , flag noDnsCache "--no-dns-cache"
        , opt restrictFileNames "--restrict-file-names"
        , flag ignoreCase "--ignore-case"
        , opt opt4 "-4"
        , opt opt6 "-6"
        , opt preferFamily "--prefer-family"
        , opt user "--user"
        , opt password "--password"
        , flag askPassword "--ask-password"
        , opt useAskpass "--use-askpass"
        , flag noIri "--no-iri"
        , opt localEncoding "--local-encoding"
        , opt remoteEncoding "--remote-encoding"
        , flag unlink "--unlink"
        , flag xattr "--xattr"
        , opt optX "-x"
        , flag protocolDirectories "--protocol-directories"
        , opt optP "-P"
        , optShow cutDirs "--cut-dirs"
        , opt httpUser "--http-user"
        , opt httpPassword "--http-password"
        , flag noCache "--no-cache"
        , opt defaultPage "--default-page"
        , flag ignoreLength "--ignore-length"
        , opt header "--header"
        , opt compression "--compression"
        , flag maxRedirect "--max-redirect"
        , opt proxyUser "--proxy-user"
        , opt proxyPassword "--proxy-password"
        , opt referer "--referer"
        , flag saveHeaders "--save-headers"
        , opt optU "-U"
        , flag noHttpKeepAlive "--no-http-keep-alive"
        , flag noCookies "--no-cookies"
        , optShow loadCookies "--load-cookies"
        , optShow saveCookies "--save-cookies"
        , flag keepSessionCookies "--keep-session-cookies"
        , opt postData "--post-data"
        , optShow postFile "--post-file"
        , opt method "--method"
        , opt bodyData "--body-data"
        , optShow bodyFile "--body-file"
        , flag contentDisposition "--content-disposition"
        , flag contentOnError "--content-on-error"
        , flag authNoChallenge "--auth-no-challenge"
        , opt secureProtocol "--secure-protocol"
        , flag httpsOnly "--https-only"
        , flag noCheckCertificate "--no-check-certificate"
        , optShow certificate "--certificate"
        , opt certificateType "--certificate-type"
        , optShow privateKey "--private-key"
        , opt privateKeyType "--private-key-type"
        , optShow caCertificate "--ca-certificate"
        , optShow caDirectory "--ca-directory"
        , optShow crlFile "--crl-file"
        , optShow pinnedpubkey "--pinnedpubkey"
        , optShow randomFile "--random-file"
        , opt ciphers "--ciphers"
        , flag noHsts "--no-hsts"
        , flag hstsFile "--hsts-file"
        , opt ftpUser "--ftp-user"
        , opt ftpPassword "--ftp-password"
        , flag noRemoveListing "--no-remove-listing"
        , flag noGlob "--no-glob"
        , flag noPassiveFtp "--no-passive-ftp"
        , flag preservePermissions "--preserve-permissions"
        , flag retrSymlinks "--retr-symlinks"
        , flag ftpsImplicit "--ftps-implicit"
        , flag ftpsResumeSsl "--ftps-resume-ssl"
        , flag ftpsClearDataConnection "--ftps-clear-data-connection"
        , flag ftpsFallbackToFtp "--ftps-fallback-to-ftp"
        , opt warcFile "--warc-file"
        , opt warcHeader "--warc-header"
        , optShow warcMaxSize "--warc-max-size"
        , flag warcCdx "--warc-cdx"
        , opt warcDedup "--warc-dedup"
        , flag noWarcCompression "--no-warc-compression"
        , flag noWarcDigests "--no-warc-digests"
        , flag noWarcKeepLog "--no-warc-keep-log"
        , optShow warcTempdir "--warc-tempdir"
        , opt optR "-r"
        , opt optL "-l"
        , flag deleteAfter "--delete-after"
        , opt optK "-k"
        , flag convertFileOnly "--convert-file-only"
        , optShow backups "--backups"
        , opt optM "-m"
        , flag strictComments "--strict-comments"
        , opt acceptRegex "--accept-regex"
        , opt rejectRegex "--reject-regex"
        , opt regexType "--regex-type"
        , opt excludeDomains "--exclude-domains"
        , flag followFtp "--follow-ftp"
        , opt followTags "--follow-tags"
        , opt ignoreTags "--ignore-tags"
        , flag trustServerNames "--trust-server-names"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run wget with options and additional arguments
wget :: Options -> [Text] -> Sh Text
wget opts args = run "wget" (buildArgs opts ++ args)

-- | Run wget, ignoring output
wget_ :: Options -> [Text] -> Sh ()
wget_ opts args = run_ "wget" (buildArgs opts ++ args)
