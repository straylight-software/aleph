{-# LANGUAGE OverloadedStrings #-}

{- |
List all NVIDIA GPUs with IOMMU group and driver info.

Usage: vfio-list

Shows PCI address, description, IOMMU group, and current driver
for each NVIDIA GPU in the system.
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Vfio as Vfio
import Control.Monad (forM_)
import qualified Data.List as L

main :: IO ()
main = script $ do
    echo "═══════════════════════════════════════════════════════════"
    echo " Available NVIDIA GPUs"
    echo "═══════════════════════════════════════════════════════════"

    gpus <- Vfio.listNvidiaGpus

    if L.null gpus
        then echo "\n  No NVIDIA GPUs found.\n"
        else forM_ gpus $ \gpu -> do
            echo ""
            echo $ "  " <> Vfio.pciAddr gpu
            case Vfio.pciDesc gpu of
                Just desc -> echo $ "    " <> desc
                Nothing -> pure ()
            echo $ "    IOMMU Group: " <> fromMaybe "N/A" (Vfio.pciIommu gpu)
            echo $ "    Driver: " <> fromMaybe "none" (Vfio.pciDriver gpu)

    echo ""
