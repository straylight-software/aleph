{-# LANGUAGE OverloadedStrings #-}

{- | mdspan package definition - Kokkos reference implementation of C++23 std::mdspan.

P0009 mdspan - multidimensional array view for C++23.
GCC 15 doesn't ship it yet, so we use the Kokkos reference impl.

= Usage from Nix

@
let
  wasm = builtins.wasm ./packages.wasm;
  mdspanSpec = wasm "mdspan" {};
in
  straylight-lib.packages.build mdspanSpec
@

= CA Soundness

This package definition is sound under content-addressed derivations:

- Source hash is explicit (sha256Hash)
- No implicit dependencies
- Build configuration is complete
- Shim header is deterministic (included in postInstall)
-}
module Aleph.Nix.Packages.Mdspan (
    mdspan,
    mdspanVersion,
) where

import Aleph.Nix.Derivation
import Data.Text (Text)
import qualified Data.Text as T

-- | Current version of mdspan we package.
mdspanVersion :: Text
mdspanVersion = "0.6.0"

{- | C++23 @<mdspan>@ shim header.

The Kokkos implementation puts everything in @std::experimental::@,
but C++23 specifies @std::@. This shim aliases them.
-}
mdspanShim :: Text
mdspanShim =
    T.unlines
        [ "#pragma once"
        , "#include <experimental/mdspan>"
        , ""
        , "namespace std {"
        , "  using experimental::mdspan;"
        , "  using experimental::extents;"
        , "  using experimental::dextents;"
        , "  using experimental::layout_right;"
        , "  using experimental::layout_left;"
        , "  using experimental::layout_stride;"
        , "  using experimental::default_accessor;"
        , "  using experimental::full_extent;"
        , "  using experimental::submdspan;"
        , "}"
        ]

{- | mdspan: C++23 std::mdspan reference implementation from Kokkos.

Features:
  - Non-owning multidimensional array views
  - Compile-time and runtime extents
  - Multiple layout policies (row-major, column-major, strided)
  - Zero-overhead abstraction

Note: Header-only library with CMake for installation.
We add a shim header to expose types in @std::@ namespace.
-}
mdspan :: Drv
mdspan =
    defaultDrv
        { drvName = "mdspan"
        , drvVersion = mdspanVersion
        , drvSrc =
            SrcGitHub
                FetchGitHub
                    { ghOwner = "kokkos"
                    , ghRepo = "mdspan"
                    , ghRev = "mdspan-" <> mdspanVersion
                    , ghHash = sha256Hash "bwE+NO/n9XsWOp3GjgLHz3s0JR0CzNDernfLHVqU9Z8="
                    }
        , drvBuilder =
            CMake
                { cmakeFlags =
                    [ "-DMDSPAN_ENABLE_TESTS=OFF"
                    , "-DMDSPAN_ENABLE_EXAMPLES=OFF"
                    , "-DMDSPAN_ENABLE_BENCHMARKS=OFF"
                    ]
                , cmakeBuildType = Nothing
                }
        , drvDeps =
            emptyDeps
                { nativeBuildInputs = ["cmake"]
                }
        , drvMeta =
            Meta
                { description = "Reference implementation of P0009 std::mdspan"
                , homepage = Just "https://github.com/kokkos/mdspan"
                , license = "asl20" -- Also BSD-3, but we pick primary
                , platforms = [] -- all platforms
                , mainProgram = Nothing
                }
        , drvPhases =
            emptyPhases
                { postInstall = [WriteFile "include/mdspan" mdspanShim]
                }
        , drvStrictDeps = True
        , drvDoCheck = False
        , drvSystem = Nothing
        }
