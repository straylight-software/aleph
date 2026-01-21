-- | Check: no shared libraries

import Aleph.Build

main :: IO ()
main = do
  ctx <- getCtx
  noSharedLibs ctx
