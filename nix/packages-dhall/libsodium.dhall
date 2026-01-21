-- libsodium: modern crypto library

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "libsodium"
          , version = "1.0.20"
          , src =
              Drv.github
                "jedisct1"
                "libsodium"
                "1.0.20-RELEASE"
                "sha256-bvAAoSXEguYPmp2k9jVSTIv/cJs2Xxocc3Xsayge+ws="
          , deps = [ "gnumake" ]
          , build =
              Drv.autotools
                (   Drv.Autotools.defaults
                  //  { configureFlags = [ "--disable-pie" ]
                      , linkage = Drv.Linkage.Static
                      }
                )
          , host
          , checks = [ "std/no-shared-libs" ]
          }
