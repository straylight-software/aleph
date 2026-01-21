-- nlohmann-json: JSON for modern C++ (header-only)

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "nlohmann-json"
          , version = "3.11.3"
          , src =
              Drv.github
                "nlohmann"
                "json"
                "v3.11.3"
                "sha256-7F0Jon+1oWL7uqet5i1IgHX0fUw/+z0QwEcA3zs5xHg="
          , build = Drv.headerOnly "include"
          , host
          }
