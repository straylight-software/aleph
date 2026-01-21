-- catch2: C++ testing framework

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "catch2"
          , version = "3.8.0"
          , src =
              Drv.github
                "catchorg"
                "Catch2"
                "v3.8.0"
                "sha256-jmaQvxMRi8vP9SLhJQPrC3I3XDvAut/xq9OWedyxYmA="
          , deps = [ "cmake", "ninja" ]
          , build =
              Drv.cmake
                (   Drv.CMake.defaults
                  //  { flags =
                          [ "-DCATCH_INSTALL_DOCS=OFF", "-DCATCH_BUILD_TESTING=OFF" ]
                      , linkage = Drv.Linkage.Static
                      }
                )
          , host
          }
