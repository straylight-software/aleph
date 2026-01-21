-- Drv/Meson.dhall
-- Meson build configuration.

let F = ./Flags.dhall

let Meson =
      { options : List { mapKey : Text, mapValue : Text }
      , buildType : F.BuildType
      , linkage : F.Linkage
      }

let defaults
    : Meson
    = { options = [] : List { mapKey : Text, mapValue : Text }
      , buildType = F.BuildType.Release
      , linkage = F.Linkage.Static
      }

in  { Meson, defaults }
