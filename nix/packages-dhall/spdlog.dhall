-- spdlog: fast C++ logging library

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "spdlog"
          , version = "1.15.0"
          , src =
              Drv.github
                "gabime"
                "spdlog"
                "v1.15.0"
                "sha256-sL+vcHrieXUrT9FUoP1c3vVwv1I3vbPkJo2uxEhHqAc="
          , deps = [ "cmake", "ninja", "fmt" ]
          , build =
              Drv.cmake
                (   Drv.CMake.defaults
                  //  { flags =
                          [ "-DSPDLOG_FMT_EXTERNAL=ON"
                          , "-DSPDLOG_BUILD_SHARED=OFF"
                          , "-DSPDLOG_BUILD_EXAMPLE=OFF"
                          , "-DSPDLOG_BUILD_TESTS=OFF"
                          ]
                      , linkage = Drv.Linkage.Static
                      }
                )
          , host
          }
