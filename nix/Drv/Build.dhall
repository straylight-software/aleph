-- Drv/Build.dhall
-- Build system sum type.

let CMake = ./CMake.dhall
let Autotools = ./Autotools.dhall
let Meson = ./Meson.dhall

let Build
    : Type
    = < CMake : CMake.CMake
      | Autotools : Autotools.Autotools
      | Meson : Meson.Meson
      | HeaderOnly : { include : Text }
      | Custom : { builder : Text }
      >

let cmake = \(c : CMake.CMake) -> Build.CMake c

let autotools = \(a : Autotools.Autotools) -> Build.Autotools a

let meson = \(m : Meson.Meson) -> Build.Meson m

let headerOnly = \(dir : Text) -> Build.HeaderOnly { include = dir }

let custom = \(builder : Text) -> Build.Custom { builder }

-- Convenience: cmake with just extra flags
let cmakeWith =
      \(flags : List Text) ->
        cmake (CMake.defaults // { flags })

-- Convenience: autotools with just configure flags
let autotoolsWith =
      \(flags : List Text) ->
        autotools (Autotools.defaults // { configureFlags = flags })

in  { Build
    , cmake
    , autotools
    , meson
    , headerOnly
    , custom
    , cmakeWith
    , autotoolsWith
    , CMake
    , Autotools
    , Meson
    }
