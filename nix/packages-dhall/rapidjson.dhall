-- rapidjson: fast JSON parser (header-only)

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "rapidjson"
          , version = "1.1.0"
          , src =
              Drv.github
                "Tencent"
                "rapidjson"
                "v1.1.0"
                "sha256-SxUXSOQDZ0/3zlFI4R84J5EtyYAE3Z9qpdVFPsYvNkk="
          , build = Drv.headerOnly "include"
          , host
          }
