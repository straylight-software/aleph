-- Drv/Prelude.dhall
-- Re-export everything for convenient imports.

let Flags = ./Flags.dhall
let Triple = ./Triple.dhall
let Src = ./Src.dhall
let Build = ./Build.dhall
let Package = ./Package.dhall
let CMake = ./CMake.dhall
let Autotools = ./Autotools.dhall
let Meson = ./Meson.dhall

in  { -- Flags
      BuildType = Flags.BuildType
    , Linkage = Flags.Linkage
    , Optimization = Flags.Optimization
    , Sanitizer = Flags.Sanitizer
    , LTO = Flags.LTO
    , PIC = Flags.PIC
    , SIMD = Flags.SIMD

      -- Triple
    , Arch = Triple.Arch
    , Vendor = Triple.Vendor
    , OS = Triple.OS
    , ABI = Triple.ABI
    , Triple = Triple.Triple
    , tripleToString = Triple.toString
    , x86_64-linux-gnu = Triple.x86_64-linux-gnu
    , x86_64-linux-musl = Triple.x86_64-linux-musl
    , aarch64-linux-gnu = Triple.aarch64-linux-gnu
    , aarch64-linux-musl = Triple.aarch64-linux-musl
    , x86_64-darwin = Triple.x86_64-darwin
    , aarch64-darwin = Triple.aarch64-darwin
    , wasm32-wasi = Triple.wasm32-wasi

      -- Source
    , Src = Src.Src
    , github = Src.github
    , url = Src.url
    , local = Src.local
    , noSrc = Src.none

      -- Build
    , Build = Build.Build
    , cmake = Build.cmake
    , autotools = Build.autotools
    , meson = Build.meson
    , headerOnly = Build.headerOnly
    , custom = Build.custom
    , cmakeWith = Build.cmakeWith
    , autotoolsWith = Build.autotoolsWith
    , CMake = Build.CMake
    , Autotools = Build.Autotools
    , Meson = Build.Meson

      -- Package
    , Package = Package.Package
    , defaults = Package.defaults
    }
