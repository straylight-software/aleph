-- | Standard Meson builder

import Aleph.Build

main :: IO ()
main = do
  ctx <- getCtx
  meson ctx []
  mesonCompile ctx
  mesonInstall ctx
