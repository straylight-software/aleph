-- Drv/Autotools.dhall
-- Autotools build configuration.

let F = ./Flags.dhall

let Autotools =
      { configureFlags : List Text
      , linkage : F.Linkage
      , pic : F.PIC
      }

let defaults
    : Autotools
    = { configureFlags = [] : List Text
      , linkage = F.Linkage.Static
      , pic = F.PIC.Default
      }

in  { Autotools, defaults }
