{-# LANGUAGE OverloadedStrings #-}

{- | Typed build tools for Nix derivation phases.

This module re-exports all typed tool APIs. Use qualified imports for clarity:

@
import qualified Aleph.Nix.Tools as Tools
import qualified Aleph.Nix.Tools.Jq as Jq
import qualified Aleph.Nix.Tools.PatchElf as PatchElf
import qualified Aleph.Nix.Tools.Install as Install
import qualified Aleph.Nix.Tools.Substitute as Sub

postInstall
    [ Jq.query Jq.defaults { Jq.rawOutput = True } ".name" "package.json"
    , PatchElf.setRpath "bin/myapp" [PatchElf.rpathOut "lib"]
    , Install.bin "build/myapp" "bin/myapp"
    , Sub.inPlace "config.h" [Sub.replace "@PREFIX@" "$out"]
    ]
@

Tool dependencies are automatically tracked - no manual nativeBuildInputs needed.
-}
module Aleph.Nix.Tools (
    -- * Tool modules
    module Aleph.Nix.Tools.Jq,
    module Aleph.Nix.Tools.PatchElf,
    module Aleph.Nix.Tools.Install,
    module Aleph.Nix.Tools.Substitute,
) where

import Aleph.Nix.Tools.Install
import Aleph.Nix.Tools.Jq
import Aleph.Nix.Tools.PatchElf
import Aleph.Nix.Tools.Substitute
