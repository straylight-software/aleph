-- Drv/Triple.dhall
-- Real target triples. Sum types, not strings.

let Arch = < x86_64 | aarch64 | armv7 | riscv64 | wasm32 | powerpc64le >

let Vendor = < unknown | apple | pc | w64 | nvidia >

let OS = < linux | darwin | windows | wasi | none >

let ABI = < gnu | musl | msvc | eabi | android | none >

let Triple = { arch : Arch, vendor : Vendor, os : OS, abi : ABI }

let archToText =
      \(a : Arch) ->
        merge
          { x86_64 = "x86_64"
          , aarch64 = "aarch64"
          , armv7 = "armv7"
          , riscv64 = "riscv64"
          , wasm32 = "wasm32"
          , powerpc64le = "powerpc64le"
          }
          a

let vendorToText =
      \(v : Vendor) ->
        merge
          { unknown = "unknown"
          , apple = "apple"
          , pc = "pc"
          , w64 = "w64"
          , nvidia = "nvidia"
          }
          v

let osToText =
      \(o : OS) ->
        merge
          { linux = "linux"
          , darwin = "darwin"
          , windows = "windows"
          , wasi = "wasi"
          , none = "none"
          }
          o

let abiToText =
      \(a : ABI) ->
        merge
          { gnu = "gnu"
          , musl = "musl"
          , msvc = "msvc"
          , eabi = "eabi"
          , android = "android"
          , none = ""
          }
          a

let toString =
      \(t : Triple) ->
        let base = "${archToText t.arch}-${vendorToText t.vendor}-${osToText t.os}"
        in  merge
              { gnu = "${base}-gnu"
              , musl = "${base}-musl"
              , msvc = "${base}-msvc"
              , eabi = "${base}-eabi"
              , android = "${base}-android"
              , none = base
              }
              t.abi

-- Standard triples
let x86_64-linux-gnu =
      { arch = Arch.x86_64
      , vendor = Vendor.unknown
      , os = OS.linux
      , abi = ABI.gnu
      }

let x86_64-linux-musl =
      { arch = Arch.x86_64
      , vendor = Vendor.unknown
      , os = OS.linux
      , abi = ABI.musl
      }

let aarch64-linux-gnu =
      { arch = Arch.aarch64
      , vendor = Vendor.unknown
      , os = OS.linux
      , abi = ABI.gnu
      }

let aarch64-linux-musl =
      { arch = Arch.aarch64
      , vendor = Vendor.unknown
      , os = OS.linux
      , abi = ABI.musl
      }

let x86_64-darwin =
      { arch = Arch.x86_64
      , vendor = Vendor.apple
      , os = OS.darwin
      , abi = ABI.none
      }

let aarch64-darwin =
      { arch = Arch.aarch64
      , vendor = Vendor.apple
      , os = OS.darwin
      , abi = ABI.none
      }

let wasm32-wasi =
      { arch = Arch.wasm32
      , vendor = Vendor.unknown
      , os = OS.wasi
      , abi = ABI.none
      }

in  { Arch
    , Vendor
    , OS
    , ABI
    , Triple
    , toString
    , archToText
    , vendorToText
    , osToText
    , abiToText
    , x86_64-linux-gnu
    , x86_64-linux-musl
    , aarch64-linux-gnu
    , aarch64-linux-musl
    , x86_64-darwin
    , aarch64-darwin
    , wasm32-wasi
    }
