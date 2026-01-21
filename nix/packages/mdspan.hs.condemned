{-# LANGUAGE OverloadedStrings #-}

-- | mdspan - C++23 std::mdspan reference implementation
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "mdspan"
        , version = "0.6.0"
        , specSrc =
            SrcGitHub
                GitHubSrc
                    { ghOwner = "kokkos"
                    , ghRepo = "mdspan"
                    , ghRev = "mdspan-0.6.0"
                    , ghHash = "sha256-fvlm0jlEIJn+XKnOGqMTeY3pjqZUHDA7N0dOPyxBSaw="
                    }
        , deps = [buildDep "cmake"]
        , phases =
            emptyPhases
                { configure =
                    [ CMakeConfigure
                        (RefSrc Nothing)
                        (RefRel "build")
                        (RefOut "out" Nothing)
                        "Release"
                        ["-DMDSPAN_ENABLE_TESTS=OFF"]
                        Ninja
                    ]
                , build = [CMakeBuild (RefRel "build") Nothing Nothing]
                , install = [CMakeInstall (RefRel "build")]
                , -- Write C++23 shim header
                  fixup =
                    [ Write
                        (RefOut "out" (Just "include/mdspan"))
                        "// C++23 shim - include the reference implementation\n\
                        \#pragma once\n\
                        \#include <experimental/mdspan>\n\
                        \namespace std {\n\
                        \  using experimental::mdspan;\n\
                        \  using experimental::extents;\n\
                        \  using experimental::dextents;\n\
                        \  using experimental::layout_right;\n\
                        \  using experimental::layout_left;\n\
                        \  using experimental::layout_stride;\n\
                        \  using experimental::default_accessor;\n\
                        \}\n"
                    ]
                }
        , meta =
            Meta
                { description = "C++23 std::mdspan reference implementation"
                , homepage = Just "https://github.com/kokkos/mdspan"
                , license = "bsd3"
                , maintainers = []
                , platforms = []
                }
        }
