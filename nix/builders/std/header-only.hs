-- | Header-only library builder

import Aleph.Build
import System.Environment (getEnv)

main :: IO ()
main = do
  ctx <- getCtx
  includeDir <- getEnv "ALEPH_INCLUDE_DIR"  -- from Dhall spec
  mkdir (outPath ctx "include")
  cp (srcPath ctx includeDir) (outPath ctx "include")
