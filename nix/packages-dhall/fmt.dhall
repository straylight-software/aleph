-- fmt: modern C++ formatting library

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "fmt"
          , version = "11.1.4"
          , src =
              Drv.github
                "fmtlib"
                "fmt"
                "11.1.4"
                "sha256-E5/K6xOrVPcxPJuGUjHumghvT2o67BVEpw9mYYGetEs="
          , deps = [ "cmake", "ninja" ]
          , build =
              Drv.cmake
                (   Drv.CMake.defaults
                  //  { flags = [ "-DFMT_TEST=OFF", "-DFMT_DOC=OFF" ]
                      , linkage = Drv.Linkage.Static
                      }
                )
          , host
          }
