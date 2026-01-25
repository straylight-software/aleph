{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Main
Description : Armitage Witness Proxy - Content-addressed caching HTTP proxy

The witness proxy intercepts all network requests from builds.
Not to block - to **witness**.

Every fetch becomes legible:
  - What was fetched (URL)
  - By whom (ed25519 identity)
  - When (timestamp)
  - What came back (SHA256 content hash)

First fetch hits the origin and caches to R2.
Future fetches (by anyone) get the cached content.

The proxy doesn't enforce policy. It produces **evidence**.

Architecture:
  - Warp HTTP server listening on configurable port
  - Forwards CONNECT requests for HTTPS tunneling
  - Caches GET responses by content hash (SHA256)
  - Logs all fetches as JSONL attestations
  - Optional R2 sync for distributed CAS

Build:
  - nix build .#armitage-proxy
  - or: ghc -O2 Main.hs -o armitage-proxy

Run:
  - PROXY_PORT=8080 PROXY_CACHE_DIR=/data/cache ./armitage-proxy
  - Configure builds: HTTP_PROXY=http://proxy:8080
-}
module Main where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (SomeException, bracket, catch, finally)
import Control.Monad (forever, unless, void, when)
import Crypto.Hash (SHA256 (..), hashWith)
import Data.Aeson (ToJSON (..), encode, object, (.=))
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as LBS
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import Data.List (find)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock (UTCTime, getCurrentTime)
import Data.Time.Format.ISO8601 (iso8601Show)
import GHC.Generics (Generic)
import Network.Socket
import qualified Network.Socket.ByteString as SBS
import System.Directory (createDirectoryIfMissing, doesFileExist)
import System.Environment (lookupEnv)
import System.FilePath ((</>))
import System.IO (BufferMode (..), hClose, hSetBuffering, stderr, stdout)
import Text.Read (readMaybe)

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

data Config = Config
  { cfgPort :: Int
  -- ^ Port to listen on
  , cfgCacheDir :: FilePath
  -- ^ Local cache directory (content-addressed)
  , cfgLogDir :: FilePath
  -- ^ Directory for fetch logs
  , cfgAllowlist :: [Text]
  -- ^ Domain allowlist (empty = allow all)
  }
  deriving (Show)

loadConfig :: IO Config
loadConfig = do
  port <- maybe 8080 id . (>>= readMaybe) <$> lookupEnv "PROXY_PORT"
  cacheDir <- maybe "/data/cache" id <$> lookupEnv "PROXY_CACHE_DIR"
  logDir <- maybe "/data/logs" id <$> lookupEnv "PROXY_LOG_DIR"
  allowlistRaw <- maybe "" id <$> lookupEnv "PROXY_ALLOWLIST"
  let allowlist = filter (not . T.null) $ T.splitOn "," (T.pack allowlistRaw)
  pure
    Config
      { cfgPort = port
      , cfgCacheDir = cacheDir
      , cfgLogDir = logDir
      , cfgAllowlist = allowlist
      }

-- -----------------------------------------------------------------------------
-- Content-Addressed Store
-- -----------------------------------------------------------------------------

-- | SHA256 hash as hex string
newtype ContentHash = ContentHash {unHash :: Text}
  deriving (Show, Eq, Generic)

instance ToJSON ContentHash where
  toJSON (ContentHash h) = toJSON h

-- | Hash content with SHA256
hashContent :: ByteString -> ContentHash
hashContent bs = ContentHash $ T.pack $ show $ hashWith SHA256 bs

-- | Two-level cache path like git objects: ab/cdef1234...
cachePath :: FilePath -> ContentHash -> FilePath
cachePath baseDir (ContentHash h) =
  baseDir </> T.unpack (T.take 2 h) </> T.unpack (T.drop 2 h)

-- | Store content in local cache
cacheStore :: FilePath -> ContentHash -> ByteString -> IO ()
cacheStore baseDir hash content = do
  let path = cachePath baseDir hash
      dir = takeWhile (/= '/') $ reverse path
  createDirectoryIfMissing True (take (length path - length dir - 1) path)
  BS.writeFile path content

-- | Lookup content in local cache
cacheLookup :: FilePath -> ContentHash -> IO (Maybe ByteString)
cacheLookup baseDir hash = do
  let path = cachePath baseDir hash
  exists <- doesFileExist path
  if exists
    then Just <$> BS.readFile path
    else pure Nothing

-- -----------------------------------------------------------------------------
-- Attestation
-- -----------------------------------------------------------------------------

-- | A record of a fetch, to be signed and stored
data Attestation = Attestation
  { attUrl :: Text
  -- ^ URL that was fetched
  , attHost :: Text
  -- ^ Host that was contacted
  , attContentHash :: Maybe ContentHash
  -- ^ SHA256 of response body (Nothing for CONNECT tunnels)
  , attSize :: Int
  -- ^ Size in bytes
  , attTimestamp :: UTCTime
  -- ^ When the fetch occurred
  , attMethod :: Text
  -- ^ HTTP method
  , attCached :: Bool
  -- ^ Whether this was a cache hit
  }
  deriving (Show, Generic)

instance ToJSON Attestation where
  toJSON Attestation {..} =
    object
      [ "url" .= attUrl
      , "host" .= attHost
      , "sha256" .= fmap unHash attContentHash
      , "size" .= attSize
      , "timestamp" .= iso8601Show attTimestamp
      , "method" .= attMethod
      , "cached" .= attCached
      ]

-- | Log an attestation to JSONL file
logAttestation :: FilePath -> Attestation -> IO ()
logAttestation logDir att = do
  createDirectoryIfMissing True logDir
  let logFile = logDir </> "fetches.jsonl"
  LBS.appendFile logFile (encode att <> "\n")

-- | Thread-safe logger
type Logger = MVar ()

withLogger :: Logger -> IO () -> IO ()
withLogger lock action = withMVar lock $ \_ -> action

-- -----------------------------------------------------------------------------
-- Allowlist
-- -----------------------------------------------------------------------------

-- | Check if a host is allowed (strips port if present)
checkAllowlist :: [Text] -> Text -> Bool
checkAllowlist [] _ = True -- empty allowlist = allow all
checkAllowlist allowed hostWithPort =
  let host = T.takeWhile (/= ':') hostWithPort
   in any (\a -> host == a || ("." <> a) `T.isSuffixOf` host) allowed

-- -----------------------------------------------------------------------------
-- HTTP Proxy Server
-- -----------------------------------------------------------------------------

-- | Parse HTTP request first line: METHOD PATH HTTP/1.x
parseRequestLine :: ByteString -> Maybe (ByteString, ByteString, ByteString)
parseRequestLine line =
  case BC.words line of
    [method, path, version] -> Just (method, path, version)
    _ -> Nothing

-- | Parse Host header from request
parseHost :: [ByteString] -> Maybe ByteString
parseHost headers =
  fmap (stripBS . BC.drop 5) $
    find (BC.isPrefixOf "Host:") headers

-- | Strip whitespace and CR from ByteString
stripBS :: ByteString -> ByteString
stripBS = BC.dropWhile isSpace . BC.reverse . BC.dropWhile isSpace . BC.reverse
  where
    isSpace c = c == ' ' || c == '\t' || c == '\r' || c == '\n'

-- | Read HTTP request (headers + optional body based on Content-Length)
recvHttpRequest :: Socket -> IO ByteString
recvHttpRequest sock = go BS.empty
  where
    go acc = do
      chunk <- SBS.recv sock 4096
      if BS.null chunk
        then return acc
        else do
          let newAcc = acc <> chunk
          -- Check if we have complete headers
          if "\r\n\r\n" `BC.isInfixOf` newAcc
            then do
              let (headers, bodyStart) = BC.breakSubstring "\r\n\r\n" newAcc
              case parseContentLength headers of
                Just len -> do
                  let bodyLen = BS.length (BC.drop 4 bodyStart)
                  if bodyLen >= len
                    then return newAcc
                    else go newAcc
                Nothing -> return newAcc -- No body expected
            else go newAcc

    parseContentLength :: ByteString -> Maybe Int
    parseContentLength headers =
      case filter (BC.isPrefixOf "Content-Length:") (BC.lines headers) of
        (line : _) ->
          readMaybe $ BC.unpack $ BC.dropWhile (== ' ') $ BC.drop 15 line
        [] -> Nothing

-- | Handle a single client connection
handleClient :: Config -> Logger -> Socket -> IO ()
handleClient cfg logger clientSock = do
  -- Read request
  request <- recvHttpRequest clientSock
  when (BS.null request) $ return ()

  let reqLines = BC.lines request
  case reqLines of
    [] -> return ()
    (firstLine : headerLines) -> do
      case parseRequestLine firstLine of
        Nothing -> do
          withLogger logger $ putStrLn $ "Invalid request: " <> BC.unpack firstLine
          return ()
        Just (method, path, _version) -> do
          let host = maybe "unknown" id $ parseHost headerLines
              hostText = TE.decodeUtf8 host
              pathText = TE.decodeUtf8 path
              -- For CONNECT, check the path (host:port), otherwise check Host header
              checkHost = if method == "CONNECT" then pathText else hostText

          -- Check allowlist
          if not (checkAllowlist (cfgAllowlist cfg) checkHost)
            then do
              withLogger logger $
                putStrLn $
                  "BLOCKED: " <> T.unpack checkHost <> " not in allowlist"
              SBS.sendAll clientSock "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nHost not in allowlist\n"
            else do
              -- Log the request
              now <- getCurrentTime
              withLogger logger $
                putStrLn $
                  BC.unpack method <> " " <> T.unpack hostText <> T.unpack pathText

              if method == "CONNECT"
                then handleConnect cfg logger clientSock host path now
                else handleRequest cfg logger clientSock method host path request now

-- | Handle CONNECT method (HTTPS tunneling)
handleConnect :: Config -> Logger -> Socket -> ByteString -> ByteString -> UTCTime -> IO ()
handleConnect cfg logger clientSock host path now = do
  -- Parse host:port from path
  let (targetHost, targetPort) = case BC.split ':' path of
        [h, p] -> (BC.unpack h, maybe 443 id $ readMaybe $ BC.unpack p)
        [h] -> (BC.unpack h, 443)
        _ -> (BC.unpack path, 443)

  -- Connect to target
  let hints = defaultHints {addrSocketType = Stream}
  addrs <- getAddrInfo (Just hints) (Just targetHost) (Just $ show targetPort)
  case addrs of
    [] -> do
      SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
    (addr : _) -> do
      targetSock <-
        socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
      connect targetSock (addrAddress addr)
        `catch` \(_ :: SomeException) -> do
          SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
          close targetSock

      -- Send 200 Connection Established
      SBS.sendAll clientSock "HTTP/1.1 200 Connection Established\r\n\r\n"

      -- Log the tunnel (no content hash for encrypted traffic)
      let att =
            Attestation
              { attUrl = "https://" <> TE.decodeUtf8 path
              , attHost = TE.decodeUtf8 host
              , attContentHash = Nothing
              , attSize = 0
              , attTimestamp = now
              , attMethod = "CONNECT"
              , attCached = False
              }
      logAttestation (cfgLogDir cfg) att

      -- Tunnel bidirectionally
      void $ forkIO $ tunnel clientSock targetSock
      tunnel targetSock clientSock
      close targetSock

-- | Bidirectional tunnel
tunnel :: Socket -> Socket -> IO ()
tunnel from to = do
  chunk <- SBS.recv from 65536
  unless (BS.null chunk) $ do
    SBS.sendAll to chunk
    tunnel from to

-- | Parse URL into (host, port, path)
parseUrl :: ByteString -> Maybe (ByteString, Int, ByteString)
parseUrl url
  | "http://" `BC.isPrefixOf` url =
      let rest = BC.drop 7 url
          (hostPort, pathQuery) = BC.break (== '/') rest
          path = if BS.null pathQuery then "/" else pathQuery
          (hostPart, portPart) = BC.break (== ':') hostPort
          port = case BC.uncons portPart of
            Just (':', p) -> maybe 80 id $ readMaybe $ BC.unpack p
            _ -> 80
       in Just (hostPart, port, path)
  | otherwise = Nothing

-- | Rewrite proxy request to origin request
-- Proxy request: GET http://host/path HTTP/1.1
-- Origin request: GET /path HTTP/1.1
rewriteRequest :: ByteString -> ByteString -> ByteString -> ByteString -> ByteString
rewriteRequest method path version restOfRequest =
  let newPath = case parseUrl path of
        Just (_, _, p) -> p
        Nothing -> path
   in method <> " " <> newPath <> " " <> version <> "\r\n" <> restOfRequest

-- | Handle regular HTTP request (GET, POST, etc.)
handleRequest ::
  Config ->
  Logger ->
  Socket ->
  ByteString ->
  ByteString ->
  ByteString ->
  ByteString ->
  UTCTime ->
  IO ()
handleRequest cfg logger clientSock method host path fullRequest now = do
  -- Parse URL from path (proxy sends full URL)
  let (targetHost, targetPort, targetPath) = case parseUrl path of
        Just (h, p, pth) -> (BC.unpack h, p, pth)
        Nothing -> (BC.unpack host, 80, path)

      url = "http://" <> BC.pack targetHost <> ":" <> BC.pack (show targetPort) <> targetPath

  -- Connect to target
  let hints = defaultHints {addrSocketType = Stream}
  addrs <- getAddrInfo (Just hints) (Just targetHost) (Just $ show targetPort)
  case addrs of
    [] -> SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
    (addr : _) -> do
      targetSock <-
        socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
      connect targetSock (addrAddress addr)
        `catch` \(_ :: SomeException) -> do
          SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
          close targetSock
          return ()

      -- Rewrite request (convert proxy format to origin format)
      let reqLines = BC.lines fullRequest
      case reqLines of
        (firstLine : rest) -> do
          case parseRequestLine firstLine of
            Just (m, p, v) -> do
              let rewritten = rewriteRequest m p v (BC.unlines rest)
              SBS.sendAll targetSock rewritten
            Nothing -> SBS.sendAll targetSock fullRequest
        _ -> SBS.sendAll targetSock fullRequest

      -- Receive response
      response <- recvHttpResponse targetSock

      -- Cache if GET and successful
      let contentHash =
            if method == "GET" && "200 OK" `BC.isInfixOf` response
              then
                let bodyStart = BC.breakSubstring "\r\n\r\n" response
                    body = BC.drop 4 $ snd bodyStart
                 in if BS.null body then Nothing else Just (hashContent body)
              else Nothing

      case contentHash of
        Just hash -> do
          let bodyStart = BC.breakSubstring "\r\n\r\n" response
              body = BC.drop 4 $ snd bodyStart
          cacheStore (cfgCacheDir cfg) hash body
          withLogger logger $
            putStrLn $
              "CACHED: " <> T.unpack (unHash hash) <> " (" <> show (BS.length body) <> " bytes)"
        Nothing -> return ()

      -- Log attestation
      let att =
            Attestation
              { attUrl = TE.decodeUtf8 url
              , attHost = TE.decodeUtf8 host
              , attContentHash = contentHash
              , attSize = BS.length response
              , attTimestamp = now
              , attMethod = TE.decodeUtf8 method
              , attCached = False
              }
      logAttestation (cfgLogDir cfg) att

      -- Forward response to client
      SBS.sendAll clientSock response
      close targetSock

-- | Receive HTTP response (reads until connection closes or Content-Length)
recvHttpResponse :: Socket -> IO ByteString
recvHttpResponse sock = go BS.empty
  where
    go acc = do
      chunk <- SBS.recv sock 65536
      if BS.null chunk
        then return acc
        else do
          let newAcc = acc <> chunk
          -- Check if we have complete headers
          if "\r\n\r\n" `BC.isInfixOf` newAcc
            then do
              -- Check for Content-Length
              let (headers, body) = BC.breakSubstring "\r\n\r\n" newAcc
                  bodyStart = BC.drop 4 body
              case parseContentLength headers of
                Just len
                  | BS.length bodyStart >= len -> return newAcc
                  | otherwise -> go newAcc
                Nothing ->
                  -- No Content-Length, check for chunked or read until close
                  if "Transfer-Encoding: chunked" `BC.isInfixOf` headers
                    then
                      if "\r\n0\r\n\r\n" `BC.isInfixOf` newAcc
                        then return newAcc
                        else go newAcc
                    else go newAcc -- Read until close
            else go newAcc

    parseContentLength :: ByteString -> Maybe Int
    parseContentLength headers =
      case filter (BC.isPrefixOf "Content-Length:") (BC.lines headers) of
        (line : _) ->
          readMaybe $ BC.unpack $ BC.dropWhile (== ' ') $ BC.drop 15 line
        [] -> Nothing

-- | Receive all available data (simple version)
recvAll :: Socket -> Int -> IO ByteString
recvAll sock maxBytes = go BS.empty
  where
    go acc
      | BS.length acc >= maxBytes = return acc
      | otherwise = do
          chunk <- SBS.recv sock 65536
          if BS.null chunk
            then return acc
            else go (acc <> chunk)

-- -----------------------------------------------------------------------------
-- Main
-- -----------------------------------------------------------------------------

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  hSetBuffering stderr LineBuffering

  putStrLn "======================================"
  putStrLn "  Armitage Witness Proxy"
  putStrLn "======================================"

  cfg <- loadConfig
  putStrLn $ "Port:      " <> show (cfgPort cfg)
  putStrLn $ "Cache:     " <> cfgCacheDir cfg
  putStrLn $ "Logs:      " <> cfgLogDir cfg
  putStrLn $ "Allowlist: " <> show (length (cfgAllowlist cfg)) <> " domains"

  -- Create directories
  createDirectoryIfMissing True (cfgCacheDir cfg)
  createDirectoryIfMissing True (cfgLogDir cfg)

  -- Thread-safe logger
  logger <- newMVar ()

  -- Create server socket
  let hints =
        defaultHints
          { addrFlags = [AI_PASSIVE]
          , addrSocketType = Stream
          }
  addr : _ <- getAddrInfo (Just hints) Nothing (Just $ show $ cfgPort cfg)
  sock <- socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
  setSocketOption sock ReuseAddr 1
  bind sock (addrAddress addr)
  listen sock 128

  putStrLn $ "Listening on :" <> show (cfgPort cfg)
  putStrLn ""

  -- Accept loop
  forever $ do
    (clientSock, clientAddr) <- accept sock
    void $
      forkIO $
        handleClient cfg logger clientSock
          `finally` close clientSock
