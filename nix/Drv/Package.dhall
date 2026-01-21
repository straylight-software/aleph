-- Drv/Package.dhall
-- The derivation. Pure data.
-- A string is a package name.

let Src = ./Src.dhall
let Build = ./Build.dhall
let Triple = ./Triple.dhall

let Package =
      { name : Text
      , version : Text
      , src : Src.Src
      , deps : List Text
      , build : Build.Build
      , host : Triple.Triple
      , target : Optional Triple.Triple
      , checks : List Text
      }

let defaults =
      { deps = [] : List Text
      , target = None Triple.Triple
      , checks = [] : List Text
      }

in  { Package, defaults }
