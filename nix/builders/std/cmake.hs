-- | Standard CMake builder
-- One line of real work.

import Aleph.Build

main :: IO ()
main = do
  ctx <- getCtx
  cmake ctx []
  ninja ctx
