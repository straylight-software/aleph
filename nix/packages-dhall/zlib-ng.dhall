-- zlib-ng: modern zlib replacement

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "zlib-ng"
          , version = "2.2.4"
          , src =
              Drv.github
                "zlib-ng"
                "zlib-ng"
                "2.2.4"
                "sha256-xJi0xFbHBd511z8H/ra22K2T1aSWcKjkxuyuwd/kvBg="
          , deps = [ "cmake", "ninja" ]
          , build =
              Drv.cmake
                (   Drv.CMake.defaults
                  //  { flags = [ "-DZLIB_COMPAT=ON", "-DWITH_GTEST=OFF" ]
                      , linkage = Drv.Linkage.Static
                      }
                )
          , host
          }
