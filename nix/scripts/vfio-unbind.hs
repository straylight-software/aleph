{-# LANGUAGE OverloadedStrings #-}

{- |
Unbind a PCI device from vfio-pci and rescan the PCI bus.

Usage: vfio-unbind PCI_ADDR

Example: vfio-unbind 0000:01:00.0

All devices in the same IOMMU group will be unbound.
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
            Vfio.unbindFromVfio (pack addr)
            echoErr ":: Done"
        _ -> script $ do
            echoErr "Usage: vfio-unbind PCI_ADDR"
            echoErr ""
            echoErr "Example: vfio-unbind 0000:01:00.0"
            echoErr ""
            echoErr "Unbinds the device from vfio-pci and rescans the PCI bus."
            echoErr "Requires root privileges."
            exit 1
