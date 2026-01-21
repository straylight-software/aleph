-- Drv/CMake.dhall
-- CMake build configuration.

let F = ./Flags.dhall

let CMake =
      { flags : List Text
      , buildType : F.BuildType
      , linkage : F.Linkage
      , pic : F.PIC
      , lto : F.LTO
      }

let defaults
    : CMake
    = { flags = [] : List Text
      , buildType = F.BuildType.Release
      , linkage = F.Linkage.Static
      , pic = F.PIC.Default
      , lto = F.LTO.Off
      }

in  { CMake, defaults }
