-- | NVIDIA TensorRT builder
-- Redistrib: unpack, copy, patch rpaths

import Aleph.Build
import Aleph.Build.Elf (patchRpath)

main :: IO ()
main = do
  ctx <- getCtx

  -- Create output directories
  mkdir (outPath ctx "lib")
  mkdir (outPath ctx "include")

  -- Copy libraries and headers
  cp (srcPath ctx "lib") (outPath ctx "lib")
  cp (srcPath ctx "include") (outPath ctx "include")

  -- Patch rpaths on all shared libraries
  libs <- glob (outPath ctx "lib/*.so*")
  mapM_ (patchLib ctx) libs

  -- Verify
  hasFile ctx "lib/libnvinfer.so"

patchLib :: Ctx -> FilePath -> IO ()
patchLib ctx lib = do
  let rpaths =
        [ "$ORIGIN"
        , depPath ctx "nvidia-cudnn" "lib"
        , depPath ctx "nvidia-cublas" "lib"
        ]
  patchRpath lib rpaths
