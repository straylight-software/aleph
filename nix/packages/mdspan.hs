{-# LANGUAGE OverloadedStrings #-}

-- | mdspan - C++23 std::mdspan reference implementation from Kokkos
module Pkg where

import Aleph.Nix.Package
import Data.Text (Text)
import Prelude hiding (writeFile)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "mdspan"
        , version "0.6.0"
        , src $
            fetchFromGitHub
                [ owner "kokkos"
                , repo "mdspan"
                , rev "mdspan-0.6.0"
                , hash "sha256-bwE+NO/n9XsWOp3GjgLHz3s0JR0CzNDernfLHVqU9Z8="
                ]
        , nativeBuildInputs ["cmake"]
        , cmakeFlags
            [ "-DMDSPAN_ENABLE_TESTS=OFF"
            , "-DMDSPAN_ENABLE_EXAMPLES=OFF"
            , "-DMDSPAN_ENABLE_BENCHMARKS=OFF"
            ]
        , -- Add shim header for std:: namespace (Kokkos uses std::experimental::)
          postInstall
            [ writeFile "include/mdspan" mdspanShim
            ]
        , description "Reference implementation of P0009 std::mdspan"
        , homepage "https://github.com/kokkos/mdspan"
        , license "apache-2.0"
        ]

-- | C++23 <mdspan> shim header
mdspanShim :: Text
mdspanShim =
    "#pragma once\n\
    \#include <experimental/mdspan>\n\
    \\n\
    \namespace std {\n\
    \  using experimental::mdspan;\n\
    \  using experimental::extents;\n\
    \  using experimental::dextents;\n\
    \  using experimental::layout_right;\n\
    \  using experimental::layout_left;\n\
    \  using experimental::layout_stride;\n\
    \  using experimental::default_accessor;\n\
    \  using experimental::full_extent;\n\
    \  using experimental::submdspan;\n\
    \}\n"
