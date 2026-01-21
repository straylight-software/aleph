-- mdspan: C++23 multidimensional array view (header-only)

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "mdspan"
          , version = "0.6.0"
          , src =
              Drv.github
                "kokkos"
                "mdspan"
                "mdspan-0.6.0"
                "sha256-GRLX0lmJLCBYGO0LnxRxznKbuG3+PcOHNOXMWznzOIQ="
          , build = Drv.headerOnly "include"
          , host
          }
