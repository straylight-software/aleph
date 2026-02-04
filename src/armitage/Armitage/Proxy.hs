{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- |
Module      : Main
Description : Armitage Witness Proxy - MITM HTTP/HTTPS proxy with content-addressed caching

The witness proxy intercepts all network requests from builds, including HTTPS
via TLS interception (MITM). Not to block - to **witness**.

Every fetch becomes legible:
  - What was fetched (full URL including path)
  - By whom (ed25519 identity - future)
  - When (timestamp)
  - What came back (SHA256 content hash)

For HTTPS interception, the proxy:
  1. Generates a CA certificate on first run
  2. For each CONNECT request, generates a per-host certificate signed by the CA
  3. Terminates TLS with the client using the generated cert
  4. Connects to the origin server via TLS
  5. Forwards requests, caches responses, logs attestations

Clients must trust the CA certificate at $PROXY_CERT_DIR/ca.pem

Build:
  nix build .#armitage.proxy

Run:
  PROXY_PORT=8080 PROXY_CACHE_DIR=/data/cache PROXY_CERT_DIR=/data/certs ./armitage-proxy

Configure builds:
  HTTP_PROXY=http://proxy:8080
  HTTPS_PROXY=http://proxy:8080
  SSL_CERT_FILE=/data/certs/ca.pem
-}
module Armitage.Proxy (main) where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (SomeException, catch, finally, try)
import Control.Monad (forever, unless, void, when)
import Crypto.Hash (SHA256 (..), hashWith)
import Crypto.Number.Serialize (i2osp)
import Crypto.PubKey.RSA (PrivateKey (..), PublicKey (..), generate)
import qualified Crypto.PubKey.RSA.PKCS15 as PKCS15
import Data.ASN1.BinaryEncoding (DER (..))
import Data.ASN1.Encoding (decodeASN1', encodeASN1')
import Data.ASN1.Types
import Data.Aeson (ToJSON (..), encode, object, (.=))
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as LBS
import Data.Default (def)
import Data.Hourglass (DateTime (..))
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import Data.List (find)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.PEM (PEM (..), pemParseBS, pemWriteBS)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock (UTCTime (..), getCurrentTime)
import Data.Time.Format.ISO8601 (iso8601Show)
import Data.X509 hiding (HashSHA256)
import qualified Data.X509 as X509
import GHC.Generics (Generic)
import Network.Socket
import qualified Network.Socket.ByteString as SBS
import Network.TLS (Context)
import qualified Network.TLS as TLS
import Network.TLS.Extra.Cipher (ciphersuite_default)
import System.Directory (createDirectoryIfMissing, doesFileExist)
import System.Environment (lookupEnv)
import System.FilePath (takeDirectory, (</>))
import System.Hourglass (dateCurrent)
import System.IO (BufferMode (..), hSetBuffering, stderr, stdout)
import Text.Read (readMaybe)
import Time.Types (Date (..))

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

data Config = Config
    { cfgPort :: Int
    , cfgCacheDir :: FilePath
    , cfgLogDir :: FilePath
    , cfgCertDir :: FilePath
    , cfgAllowlist :: [Text]
    }
    deriving (Show)

loadConfig :: IO Config
loadConfig = do
    cfgPort <- maybe 8080 id . (>>= readMaybe) <$> lookupEnv "PROXY_PORT"
    cfgCacheDir <- maybe "/data/cache" id <$> lookupEnv "PROXY_CACHE_DIR"
    cfgLogDir <- maybe "/data/logs" id <$> lookupEnv "PROXY_LOG_DIR"
    cfgCertDir <- maybe "/data/certs" id <$> lookupEnv "PROXY_CERT_DIR"
    allowlistRaw <- maybe "" id <$> lookupEnv "PROXY_ALLOWLIST"
    let cfgAllowlist = filter (not . T.null) $ T.splitOn "," (T.pack allowlistRaw)
    pure Config{..}

-- -----------------------------------------------------------------------------
-- Certificate Authority
-- -----------------------------------------------------------------------------

data CA = CA
    { caCert :: SignedCertificate
    , caKey :: PrivateKey
    , caCertPEM :: ByteString
    }

-- | Generate or load CA certificate
initCA :: FilePath -> IO CA
initCA certDir = do
    createDirectoryIfMissing True certDir
    let caKeyPath = certDir </> "ca-key.pem"
        caCertPath = certDir </> "ca.pem"

    exists <- doesFileExist caCertPath
    if exists
        then loadCA caKeyPath caCertPath
        else generateCA caKeyPath caCertPath

generateCA :: FilePath -> FilePath -> IO CA
generateCA keyPath certPath = do
    putStrLn "Generating CA certificate..."

    -- Generate RSA key pair (2048 bits)
    (pubKey, privKey) <- generate 256 65537

    -- Create self-signed CA certificate
    now <- dateCurrent
    let notBefore = now
        notAfter = addYears 10 now

        dn =
            DistinguishedName
                [ (getObjectID DnCommonName, ASN1CharacterString UTF8 "Armitage Proxy CA")
                , (getObjectID DnOrganization, ASN1CharacterString UTF8 "Straylight")
                ]

        cert =
            Certificate
                { certVersion = 2
                , certSerial = 1
                , certSignatureAlg = SignatureALG X509.HashSHA256 PubKeyALG_RSA
                , certIssuerDN = dn
                , certValidity = (notBefore, notAfter)
                , certSubjectDN = dn
                , certPubKey = PubKeyRSA pubKey
                , certExtensions = Extensions $ Just [extensionEncode True basicConstraintsCA]
                }

        basicConstraintsCA = ExtBasicConstraints True (Just 0)

    signedCert <- signCertificate privKey cert

    -- Write to files
    let keyPEM = pemWriteBS $ PEM "RSA PRIVATE KEY" [] (encodePrivKey privKey)
        certPEM = pemWriteBS $ PEM "CERTIFICATE" [] (encodeSignedObject signedCert)

    BS.writeFile keyPath keyPEM
    BS.writeFile certPath certPEM
    putStrLn $ "CA certificate written to: " <> certPath

    pure CA{caCert = signedCert, caKey = privKey, caCertPEM = certPEM}

loadCA :: FilePath -> FilePath -> IO CA
loadCA keyPath certPath = do
    putStrLn "Loading existing CA certificate..."
    keyPEM <- BS.readFile keyPath
    certPEM <- BS.readFile certPath

    -- Parse the PEM-encoded private key
    case pemParseBS keyPEM of
        Left err -> do
            putStrLn $ "Failed to parse CA key PEM: " <> err
            generateCA keyPath certPath
        Right pems -> case pems of
            [] -> do
                putStrLn "No PEM blocks found in CA key file"
                generateCA keyPath certPath
            (pem : _) -> case decodePrivKey (pemContent pem) of
                Left err -> do
                    putStrLn $ "Failed to decode CA private key: " <> err
                    generateCA keyPath certPath
                Right privKey -> do
                    -- Parse the certificate
                    case pemParseBS certPEM of
                        Left err -> do
                            putStrLn $ "Failed to parse CA cert PEM: " <> err
                            generateCA keyPath certPath
                        Right certPems -> case certPems of
                            [] -> do
                                putStrLn "No PEM blocks found in CA cert file"
                                generateCA keyPath certPath
                            (certPem : _) -> case decodeSignedCertificate (pemContent certPem) of
                                Left err -> do
                                    putStrLn $ "Failed to decode CA certificate: " <> err
                                    generateCA keyPath certPath
                                Right signedCert -> do
                                    putStrLn "CA certificate loaded successfully"
                                    pure CA{caCert = signedCert, caKey = privKey, caCertPEM = certPEM}

{- | Sign a certificate with an RSA private key using SHA256

Uses objectToSignedExact from Data.X509 with PKCS#15 RSA signing
-}
signCertificate :: PrivateKey -> Certificate -> IO SignedCertificate
signCertificate privKey cert = do
    let sigAlg = SignatureALG X509.HashSHA256 PubKeyALG_RSA

        -- The signing function: takes DER-encoded TBSCertificate,
        -- returns (signature, algorithm, ())
        signFunction :: ByteString -> (ByteString, SignatureALG, ())
        signFunction tbsData =
            case PKCS15.sign Nothing (Just SHA256) privKey tbsData of
                Left err -> error $ "RSA signing failed: " <> show err
                Right sig -> (sig, sigAlg, ())

        -- Sign the certificate
        (signedCert, ()) = objectToSignedExact signFunction cert

    pure signedCert

{- | Encode RSA private key to PKCS#1 DER format

RSAPrivateKey ::= SEQUENCE {
  version           Version,
  modulus           INTEGER,  -- n
  publicExponent    INTEGER,  -- e
  privateExponent   INTEGER,  -- d
  prime1            INTEGER,  -- p
  prime2            INTEGER,  -- q
  exponent1         INTEGER,  -- d mod (p-1)
  exponent2         INTEGER,  -- d mod (q-1)
  coefficient       INTEGER   -- (inverse of q) mod p
}
-}
encodePrivKey :: PrivateKey -> ByteString
encodePrivKey pk =
    encodeASN1'
        DER
        [ Start Sequence
        , IntVal 0 -- version
        , IntVal (public_n $ private_pub pk) -- modulus
        , IntVal (public_e $ private_pub pk) -- public exponent
        , IntVal (private_d pk) -- private exponent
        , IntVal (private_p pk) -- prime1
        , IntVal (private_q pk) -- prime2
        , IntVal (private_dP pk) -- exponent1
        , IntVal (private_dQ pk) -- exponent2
        , IntVal (private_qinv pk) -- coefficient
        , End Sequence
        ]

-- | Decode RSA private key from PKCS#1 DER format
decodePrivKey :: ByteString -> Either String PrivateKey
decodePrivKey bs = do
    asn1 <- either (Left . show) Right $ decodeASN1' DER bs
    case asn1 of
        [ Start Sequence
            , IntVal _ver
            , IntVal n
            , IntVal e
            , IntVal d
            , IntVal p
            , IntVal q
            , IntVal dP
            , IntVal dQ
            , IntVal qinv
            , End Sequence
            ] ->
                Right
                    PrivateKey
                        { private_pub =
                            PublicKey
                                { public_size = (fromIntegral (BS.length (i2osp n)))
                                , public_n = n
                                , public_e = e
                                }
                        , private_d = d
                        , private_p = p
                        , private_q = q
                        , private_dP = dP
                        , private_dQ = dQ
                        , private_qinv = qinv
                        }
        _ -> Left "Invalid PKCS#1 RSA private key format"

-- | Add years to DateTime
addYears :: Int -> DateTime -> DateTime
addYears n dt@DateTime{..} =
    let Date y m d = dtDate
     in dt{dtDate = Date (y + n) m d}

-- -----------------------------------------------------------------------------
-- Per-Host Certificate Generation
-- -----------------------------------------------------------------------------

-- | Cache of generated host certificates
type CertCache = IORef (Map String (SignedCertificate, PrivateKey))

-- | Get or generate certificate for a host
getHostCert :: CA -> CertCache -> String -> IO (SignedCertificate, PrivateKey)
getHostCert ca cache host = do
    cached <- Map.lookup host <$> readIORef cache
    case cached of
        Just pair -> pure pair
        Nothing -> do
            pair <- generateHostCert ca host
            atomicModifyIORef' cache $ \m -> (Map.insert host pair m, ())
            pure pair

-- | Generate a certificate for a specific host
generateHostCert :: CA -> String -> IO (SignedCertificate, PrivateKey)
generateHostCert CA{..} host = do
    -- Generate new RSA key for this host
    (pubKey, privKey) <- generate 256 65537

    now <- dateCurrent
    let notBefore = now
        notAfter = addYears 1 now

        issuerDN = certSubjectDN $ signedObject $ getSigned caCert

        subjectDN =
            DistinguishedName
                [(getObjectID DnCommonName, ASN1CharacterString UTF8 (BC.pack host))]

        -- Subject Alternative Name for the host
        san = ExtSubjectAltName [AltNameDNS host]

        cert =
            Certificate
                { certVersion = 2
                , certSerial = 12345 -- Should be random
                , certSignatureAlg = SignatureALG X509.HashSHA256 PubKeyALG_RSA
                , certIssuerDN = issuerDN
                , certValidity = (notBefore, notAfter)
                , certSubjectDN = subjectDN
                , certPubKey = PubKeyRSA pubKey
                , certExtensions = Extensions $ Just [extensionEncode False san]
                }

    signedCert <- signCertificate caKey cert
    pure (signedCert, privKey)

-- -----------------------------------------------------------------------------
-- Content-Addressed Store
-- -----------------------------------------------------------------------------

newtype ContentHash = ContentHash {unHash :: Text}
    deriving (Show, Eq, Generic)

instance ToJSON ContentHash where
    toJSON (ContentHash h) = toJSON h

hashContent :: ByteString -> ContentHash
hashContent bs = ContentHash $ T.pack $ show $ hashWith SHA256 bs

cachePath :: FilePath -> ContentHash -> FilePath
cachePath baseDir (ContentHash h) =
    baseDir </> T.unpack (T.take 2 h) </> T.unpack (T.drop 2 h)

cacheStore :: FilePath -> ContentHash -> ByteString -> IO ()
cacheStore baseDir hash content = do
    let path = cachePath baseDir hash
        dir = takeDirectory path
    createDirectoryIfMissing True dir
    BS.writeFile path content

-- -----------------------------------------------------------------------------
-- Attestation
-- -----------------------------------------------------------------------------

data Attestation = Attestation
    { attUrl :: Text
    , attHost :: Text
    , attContentHash :: Maybe ContentHash
    , attSize :: Int
    , attTimestamp :: UTCTime
    , attMethod :: Text
    , attCached :: Bool
    }
    deriving (Show, Generic)

instance ToJSON Attestation where
    toJSON Attestation{..} =
        object
            [ "url" .= attUrl
            , "host" .= attHost
            , "sha256" .= fmap unHash attContentHash
            , "size" .= attSize
            , "timestamp" .= iso8601Show attTimestamp
            , "method" .= attMethod
            , "cached" .= attCached
            ]

logAttestation :: FilePath -> Attestation -> IO ()
logAttestation logDir att = do
    createDirectoryIfMissing True logDir
    let logFile = logDir </> "fetches.jsonl"
    LBS.appendFile logFile (encode att <> "\n")

-- -----------------------------------------------------------------------------
-- Allowlist
-- -----------------------------------------------------------------------------

checkAllowlist :: [Text] -> Text -> Bool
checkAllowlist [] _ = True
checkAllowlist allowed hostWithPort =
    let host = T.takeWhile (/= ':') hostWithPort
     in any (\a -> host == a || ("." <> a) `T.isSuffixOf` host) allowed

-- -----------------------------------------------------------------------------
-- HTTP Parsing
-- -----------------------------------------------------------------------------

parseRequestLine :: ByteString -> Maybe (ByteString, ByteString, ByteString)
parseRequestLine line =
    case BC.words line of
        [method, path, version] -> Just (method, path, version)
        _ -> Nothing

parseHost :: [ByteString] -> Maybe ByteString
parseHost headers =
    fmap (stripBS . BC.drop 5) $
        find (BC.isPrefixOf "Host:") headers

stripBS :: ByteString -> ByteString
stripBS = BC.dropWhile isSpace . BC.reverse . BC.dropWhile isSpace . BC.reverse
  where
    isSpace c = c == ' ' || c == '\t' || c == '\r' || c == '\n'

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

-- -----------------------------------------------------------------------------
-- Logger
-- -----------------------------------------------------------------------------

type Logger = MVar ()

withLogger :: Logger -> IO () -> IO ()
withLogger lock action = withMVar lock $ \_ -> action

-- -----------------------------------------------------------------------------
-- HTTP Request/Response Handling
-- -----------------------------------------------------------------------------

recvHttpRequest :: Socket -> IO ByteString
recvHttpRequest sock = go BS.empty
  where
    go acc = do
        chunk <- SBS.recv sock 4096
        if BS.null chunk
            then return acc
            else do
                let newAcc = acc <> chunk
                if "\r\n\r\n" `BC.isInfixOf` newAcc
                    then do
                        let (headers, _) = BC.breakSubstring "\r\n\r\n" newAcc
                        case parseContentLength headers of
                            Just len -> do
                                let bodyLen = BS.length newAcc - BS.length headers - 4
                                if bodyLen >= len
                                    then return newAcc
                                    else go newAcc
                            Nothing -> return newAcc
                    else go newAcc

    parseContentLength :: ByteString -> Maybe Int
    parseContentLength headers =
        case filter (BC.isPrefixOf "Content-Length:") (BC.lines headers) of
            (line : _) ->
                readMaybe $ BC.unpack $ BC.dropWhile (== ' ') $ BC.drop 15 line
            [] -> Nothing

recvHttpResponse :: Socket -> IO ByteString
recvHttpResponse sock = go BS.empty
  where
    go acc = do
        chunk <- SBS.recv sock 65536
        if BS.null chunk
            then return acc
            else do
                let newAcc = acc <> chunk
                if "\r\n\r\n" `BC.isInfixOf` newAcc
                    then do
                        let (headers, body) = BC.breakSubstring "\r\n\r\n" newAcc
                            bodyStart = BC.drop 4 body
                        case parseContentLength headers of
                            Just len
                                | BS.length bodyStart >= len -> return newAcc
                                | otherwise -> go newAcc
                            Nothing ->
                                if "Transfer-Encoding: chunked" `BC.isInfixOf` headers
                                    then
                                        if "\r\n0\r\n\r\n" `BC.isInfixOf` newAcc
                                            then return newAcc
                                            else go newAcc
                                    else go newAcc
                    else go newAcc

    parseContentLength :: ByteString -> Maybe Int
    parseContentLength headers =
        case filter (BC.isPrefixOf "Content-Length:") (BC.lines headers) of
            (line : _) ->
                readMaybe $ BC.unpack $ BC.dropWhile (== ' ') $ BC.drop 15 line
            [] -> Nothing

-- -----------------------------------------------------------------------------
-- Proxy Handlers
-- -----------------------------------------------------------------------------

-- | Handle client connection
handleClient :: Config -> CA -> CertCache -> Logger -> Socket -> IO ()
handleClient cfg ca certCache logger clientSock = do
    request <- recvHttpRequest clientSock
    when (BS.null request) $ return ()

    let reqLines = BC.lines request
    case reqLines of
        [] -> return ()
        (firstLine : headerLines) -> do
            case parseRequestLine firstLine of
                Nothing -> do
                    withLogger logger $ putStrLn $ "Invalid request: " <> BC.unpack firstLine
                Just (method, path, _version) -> do
                    let host = maybe "unknown" id $ parseHost headerLines
                        hostText = TE.decodeUtf8 host
                        pathText = TE.decodeUtf8 path
                        checkHost = if method == "CONNECT" then pathText else hostText

                    if not (checkAllowlist (cfgAllowlist cfg) checkHost)
                        then do
                            withLogger logger $ putStrLn $ "BLOCKED: " <> T.unpack checkHost
                            SBS.sendAll clientSock "HTTP/1.1 403 Forbidden\r\n\r\nBlocked\n"
                        else do
                            now <- getCurrentTime
                            withLogger logger $ putStrLn $ BC.unpack method <> " " <> T.unpack checkHost

                            if method == "CONNECT"
                                then handleConnect cfg ca certCache logger clientSock host path now
                                else handleHttp cfg logger clientSock method host path request now

-- | Handle CONNECT (HTTPS interception via TLS MITM)
handleConnect :: Config -> CA -> CertCache -> Logger -> Socket -> ByteString -> ByteString -> UTCTime -> IO ()
handleConnect cfg ca certCache logger clientSock host path now = do
    let (targetHost, targetPort) = case BC.split ':' path of
            [h, p] -> (BC.unpack h, maybe 443 id $ readMaybe $ BC.unpack p)
            [h] -> (BC.unpack h, 443)
            _ -> (BC.unpack path, 443)

    -- Send 200 Connection Established to tell client we're ready for TLS
    SBS.sendAll clientSock "HTTP/1.1 200 Connection Established\r\n\r\n"

    -- Get/generate certificate for this host
    (hostCert, hostKey) <- getHostCert ca certCache targetHost

    -- Create TLS server context for client connection (we act as server to client)
    let serverCredential = (CertificateChain [hostCert], PrivKeyRSA hostKey)
        serverParams =
            def
                { TLS.serverShared =
                    def
                        { TLS.sharedCredentials = TLS.Credentials [serverCredential]
                        }
                , TLS.serverSupported =
                    def
                        { TLS.supportedCiphers = ciphersuite_default
                        , TLS.supportedVersions = [TLS.TLS13, TLS.TLS12]
                        }
                }

    -- Perform TLS handshake with client
    result <- try $ do
        clientCtx <- TLS.contextNew clientSock serverParams
        TLS.handshake clientCtx

        -- Connect to origin server
        let hints = defaultHints{addrSocketType = Stream}
        addrs <- getAddrInfo (Just hints) (Just targetHost) (Just $ show targetPort)
        case addrs of
            [] -> do
                TLS.sendData clientCtx "HTTP/1.1 502 Bad Gateway\r\n\r\nCannot resolve host\n"
                TLS.bye clientCtx
            (addr : _) -> do
                targetSock <- socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
                connect targetSock (addrAddress addr)

                -- Create TLS client context for origin connection (we act as client to origin)
                let clientParams =
                        (TLS.defaultParamsClient targetHost (BC.pack $ show targetPort))
                            { TLS.clientSupported =
                                def
                                    { TLS.supportedCiphers = ciphersuite_default
                                    , TLS.supportedVersions = [TLS.TLS13, TLS.TLS12]
                                    }
                            , TLS.clientShared =
                                def
                                    { -- Accept all certificates (we're proxying, not validating)
                                      TLS.sharedValidationCache =
                                        TLS.ValidationCache
                                            (\_ _ _ -> return TLS.ValidationCachePass)
                                            (\_ _ _ -> return ())
                                    }
                            }

                targetCtx <- TLS.contextNew targetSock clientParams
                TLS.handshake targetCtx

                -- Now proxy HTTP requests over both TLS connections
                proxyHttpsRequests cfg logger clientCtx targetCtx targetHost now
                    `finally` do
                        TLS.bye targetCtx `catch` \(_ :: SomeException) -> pure ()
                        close targetSock

                TLS.bye clientCtx `catch` \(_ :: SomeException) -> pure ()

    case result of
        Left (e :: SomeException) -> do
            withLogger logger $ putStrLn $ "HTTPS interception error for " <> targetHost <> ": " <> show e
            -- Fall back to simple tunnel for incompatible clients
            fallbackTunnel clientSock targetHost targetPort
        Right () -> pure ()

-- | Proxy HTTP requests over intercepted TLS connections
proxyHttpsRequests :: Config -> Logger -> Context -> Context -> String -> UTCTime -> IO ()
proxyHttpsRequests cfg logger clientCtx targetCtx targetHost startTime = do
    -- Read HTTP request from client over TLS
    requestData <- TLS.recvData clientCtx
    unless (BS.null requestData) $ do
        let reqLines = BC.lines requestData
        case reqLines of
            [] -> pure ()
            (firstLine : headerLines) -> do
                case parseRequestLine firstLine of
                    Nothing -> do
                        withLogger logger $ putStrLn $ "Invalid HTTPS request: " <> BC.unpack firstLine
                    Just (method, path, _version) -> do
                        now <- getCurrentTime
                        let hostHeader = maybe (BC.pack targetHost) id $ parseHost headerLines
                            url = "https://" <> hostHeader <> path

                        withLogger logger $
                            putStrLn $
                                "HTTPS " <> BC.unpack method <> " " <> BC.unpack url

                        -- Forward request to origin
                        TLS.sendData targetCtx (LBS.fromStrict requestData)

                        -- Read response from origin
                        responseData <- recvTlsResponse targetCtx

                        -- Cache GET responses
                        let contentHash =
                                if method == "GET" && "200 OK" `BC.isInfixOf` responseData
                                    then
                                        let (_, body) = BC.breakSubstring "\r\n\r\n" responseData
                                            bodyContent = BC.drop 4 body
                                         in if BS.null bodyContent then Nothing else Just (hashContent bodyContent)
                                    else Nothing

                        case contentHash of
                            Just hash -> do
                                let (_, body) = BC.breakSubstring "\r\n\r\n" responseData
                                    bodyContent = BC.drop 4 body
                                cacheStore (cfgCacheDir cfg) hash bodyContent
                                withLogger logger $ putStrLn $ "HTTPS CACHED: " <> T.unpack (unHash hash)
                            Nothing -> pure ()

                        -- Log attestation
                        let att =
                                Attestation
                                    { attUrl = TE.decodeUtf8 url
                                    , attHost = TE.decodeUtf8 hostHeader
                                    , attContentHash = contentHash
                                    , attSize = BS.length responseData
                                    , attTimestamp = now
                                    , attMethod = TE.decodeUtf8 method
                                    , attCached = False
                                    }
                        logAttestation (cfgLogDir cfg) att

                        -- Forward response to client
                        TLS.sendData clientCtx (LBS.fromStrict responseData)

                        -- Continue proxying (HTTP keep-alive)
                        proxyHttpsRequests cfg logger clientCtx targetCtx targetHost startTime

-- | Read HTTP response from TLS context
recvTlsResponse :: Context -> IO ByteString
recvTlsResponse ctx = go BS.empty
  where
    go acc = do
        chunk <- TLS.recvData ctx
        if BS.null chunk
            then return acc
            else do
                let newAcc = acc <> chunk
                if "\r\n\r\n" `BC.isInfixOf` newAcc
                    then do
                        let (headers, body) = BC.breakSubstring "\r\n\r\n" newAcc
                            bodyStart = BC.drop 4 body
                        case parseContentLength headers of
                            Just len
                                | BS.length bodyStart >= len -> return newAcc
                                | otherwise -> go newAcc
                            Nothing ->
                                if "Transfer-Encoding: chunked" `BC.isInfixOf` headers
                                    then
                                        if "\r\n0\r\n\r\n" `BC.isInfixOf` newAcc
                                            then return newAcc
                                            else go newAcc
                                    else return newAcc -- Assume complete for now
                    else go newAcc

    parseContentLength :: ByteString -> Maybe Int
    parseContentLength headers =
        case filter (BC.isPrefixOf "Content-Length:") (BC.lines headers) of
            (line : _) ->
                readMaybe $ BC.unpack $ BC.dropWhile (== ' ') $ BC.drop 15 line
            [] -> Nothing

-- | Fallback to simple TCP tunnel when TLS interception fails
fallbackTunnel :: Socket -> String -> Int -> IO ()
fallbackTunnel clientSock targetHost targetPort = do
    let hints = defaultHints{addrSocketType = Stream}
    addrs <- getAddrInfo (Just hints) (Just targetHost) (Just $ show targetPort)
    case addrs of
        [] -> pure ()
        (addr : _) -> do
            targetSock <- socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
            connect targetSock (addrAddress addr)
                `catch` \(_ :: SomeException) -> close targetSock

            -- Simple tunnel
            void $ forkIO $ tunnel clientSock targetSock
            tunnel targetSock clientSock
            close targetSock

tunnel :: Socket -> Socket -> IO ()
tunnel from to = do
    chunk <- SBS.recv from 65536
    unless (BS.null chunk) $ do
        SBS.sendAll to chunk
        tunnel from to

-- | Handle HTTP request
handleHttp :: Config -> Logger -> Socket -> ByteString -> ByteString -> ByteString -> ByteString -> UTCTime -> IO ()
handleHttp cfg logger clientSock method host path fullRequest now = do
    let (targetHost, targetPort, targetPath) = case parseUrl path of
            Just (h, p, pth) -> (BC.unpack h, p, pth)
            Nothing -> (BC.unpack host, 80, path)

        url = "http://" <> BC.pack targetHost <> ":" <> BC.pack (show targetPort) <> targetPath

    let hints = defaultHints{addrSocketType = Stream}
    addrs <- getAddrInfo (Just hints) (Just targetHost) (Just $ show targetPort)
    case addrs of
        [] -> SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
        (addr : _) -> do
            targetSock <- socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
            connect targetSock (addrAddress addr)
                `catch` \(_ :: SomeException) -> do
                    SBS.sendAll clientSock "HTTP/1.1 502 Bad Gateway\r\n\r\n"
                    close targetSock
                    return ()

            -- Rewrite request
            let reqLines = BC.lines fullRequest
            case reqLines of
                (firstLine : rest) -> do
                    case parseRequestLine firstLine of
                        Just (m, p, v) -> do
                            let newPath = case parseUrl p of
                                    Just (_, _, pth) -> pth
                                    Nothing -> p
                                rewritten = m <> " " <> newPath <> " " <> v <> "\r\n" <> BC.unlines rest
                            SBS.sendAll targetSock rewritten
                        Nothing -> SBS.sendAll targetSock fullRequest
                _ -> SBS.sendAll targetSock fullRequest

            response <- recvHttpResponse targetSock

            -- Cache GET responses
            let contentHash =
                    if method == "GET" && "200 OK" `BC.isInfixOf` response
                        then
                            let (_, body) = BC.breakSubstring "\r\n\r\n" response
                                bodyContent = BC.drop 4 body
                             in if BS.null bodyContent then Nothing else Just (hashContent bodyContent)
                        else Nothing

            case contentHash of
                Just hash -> do
                    let (_, body) = BC.breakSubstring "\r\n\r\n" response
                        bodyContent = BC.drop 4 body
                    cacheStore (cfgCacheDir cfg) hash bodyContent
                    withLogger logger $ putStrLn $ "CACHED: " <> T.unpack (unHash hash)
                Nothing -> return ()

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

            SBS.sendAll clientSock response
            close targetSock

-- -----------------------------------------------------------------------------
-- Main
-- -----------------------------------------------------------------------------

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    hSetBuffering stderr LineBuffering

    putStrLn "======================================"
    putStrLn "  Armitage Witness Proxy (TLS MITM)"
    putStrLn "======================================"

    cfg <- loadConfig
    putStrLn $ "Port:      " <> show (cfgPort cfg)
    putStrLn $ "Cache:     " <> cfgCacheDir cfg
    putStrLn $ "Logs:      " <> cfgLogDir cfg
    putStrLn $ "Certs:     " <> cfgCertDir cfg
    putStrLn $ "Allowlist: " <> show (length (cfgAllowlist cfg)) <> " domains"

    createDirectoryIfMissing True (cfgCacheDir cfg)
    createDirectoryIfMissing True (cfgLogDir cfg)

    -- Initialize CA
    ca <- initCA (cfgCertDir cfg)
    certCache <- newIORef Map.empty
    logger <- newMVar ()

    putStrLn ""
    putStrLn $ "Trust the CA certificate at: " <> cfgCertDir cfg </> "ca.pem"
    putStrLn ""

    -- Create server socket
    let hints = defaultHints{addrFlags = [AI_PASSIVE], addrSocketType = Stream}
    addr : _ <- getAddrInfo (Just hints) Nothing (Just $ show $ cfgPort cfg)
    sock <- socket (addrFamily addr) (addrSocketType addr) (addrProtocol addr)
    setSocketOption sock ReuseAddr 1
    bind sock (addrAddress addr)
    listen sock 128

    putStrLn $ "Listening on :" <> show (cfgPort cfg)
    putStrLn ""

    forever $ do
        (clientSock, _) <- accept sock
        void $
            forkIO $
                handleClient cfg ca certCache logger clientSock
                    `finally` close clientSock
