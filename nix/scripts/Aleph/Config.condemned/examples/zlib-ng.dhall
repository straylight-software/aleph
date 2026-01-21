-- Example: zlib-ng using the typed CA-derivation schema
--
-- Compare to the Haskell version in Aleph.Nix.Packages.ZlibNg
-- This Dhall can be:
--   1. Read by Haskell via dhall-to-json
--   2. Passed to WASM builder
--   3. Interpreted directly by Nix

let Drv = ./Drv.dhall

let zlibNg : Drv.Drv =
  Drv.defaultDrv //
    { pname = "zlib-ng"
    , version = "2.2.4"
    , system = "x86_64-linux"
    , contentAddressed = True
    
    , outputs = 
        [ Drv.floatingOut "out"
        , Drv.floatingOut "dev"
        ]
    
    , src = Drv.Src.GitHub
        { owner = "zlib-ng"
        , repo = "zlib-ng"
        , rev = "2.2.4"
        , hash = "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
        }
    
    , deps =
        [ Drv.buildDep "cmake"
        , Drv.buildDep "pkg-config"
        , Drv.buildDep "ninja"
        , Drv.checkDep "gtest"
        ]
    
    , phases = Drv.emptyPhases //
        { configure = 
            [ Drv.cmake
                [ "-DBUILD_STATIC_LIBS=ON"
                , "-DBUILD_SHARED_LIBS=OFF"
                , "-DZLIB_COMPAT=ON"
                , "-DINSTALL_UTILS=ON"
                ]
            ]
        
        , build =
            [ Drv.Action.CMakeBuild
                { buildDir = Drv.rel "build"
                , target = None Text
                , jobs = None Natural
                }
            ]
        
        , install =
            [ Drv.Action.CMakeInstall { buildDir = Drv.rel "build" }
            ]
        
        , fixup =
            -- Move headers to dev output
            [ Drv.mkdir (Drv.outNamed "dev")
            , Drv.copy (Drv.outSub "include") (Drv.Ref.Out { name = "dev", subpath = Some "include" })
            ]
        }
    
    , meta =
        { description = "zlib data compression library for the next generation systems"
        , homepage = Some "https://github.com/zlib-ng/zlib-ng"
        , license = "zlib"
        , maintainers = ["b7r6"]
        , platforms = ["x86_64-linux", "aarch64-linux"]
        }
    }

in zlibNg
