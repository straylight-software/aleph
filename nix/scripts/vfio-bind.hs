{-# LANGUAGE OverloadedStrings #-}

{- |
Bind a PCI device to vfio-pci for GPU passthrough.

Usage: vfio-bind PCI_ADDR

Example: vfio-bind 0000:01:00.0

All devices in the same IOMMU group will be bound together.
Requires root privileges.
-}
module Main where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Vfio as Vfio
import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [addr] -> script $ do
            devices <- Vfio.bindToVfio (pack addr)
            echoErr ""
            echoErr ":: Bound devices:"
            mapM_ (\d -> echoErr $ "   " <> d) devices
        _ -> script $ do
            echoErr "Usage: vfio-bind PCI_ADDR"
            echoErr ""
            echoErr "Example: vfio-bind 0000:01:00.0"
            echoErr ""
            echoErr "Binds the device and all IOMMU group members to vfio-pci."
            echoErr "Requires root privileges."
            exit 1
