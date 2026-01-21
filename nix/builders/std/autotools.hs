-- | Standard autotools builder

import Aleph.Build

main :: IO ()
main = do
  ctx <- getCtx
  configure ctx []
  make ctx []
  make ctx ["install"]
