-- Drv/Triple.dhall
-- Real target triples. Sum types, not strings.
-- Constructor names match Haskell (PascalCase) for dhall-haskell compatibility.

let Arch = < X86_64 | AArch64 | ARMv7 | RISCV64 | WASM32 | PowerPC64LE >

let Vendor = < Unknown | Apple | PC | W64 | Nvidia >

let OS = < Linux | Darwin | Windows | WASI | NoOS >

let ABI = < GNU | Musl | MSVC | EABI | Android | NoABI >

let Triple = { arch : Arch, vendor : Vendor, os : OS, abi : ABI }

let archToText =
      \(a : Arch) ->
        merge
          { X86_64 = "x86_64"
          , AArch64 = "aarch64"
          , ARMv7 = "armv7"
          , RISCV64 = "riscv64"
          , WASM32 = "wasm32"
          , PowerPC64LE = "powerpc64le"
          }
          a

let vendorToText =
      \(v : Vendor) ->
        merge
          { Unknown = "unknown"
          , Apple = "apple"
          , PC = "pc"
          , W64 = "w64"
          , Nvidia = "nvidia"
          }
          v

let osToText =
      \(o : OS) ->
        merge
          { Linux = "linux"
          , Darwin = "darwin"
          , Windows = "windows"
          , WASI = "wasi"
          , NoOS = "none"
          }
          o

let abiToText =
      \(a : ABI) ->
        merge
          { GNU = "gnu"
          , Musl = "musl"
          , MSVC = "msvc"
          , EABI = "eabi"
          , Android = "android"
          , NoABI = ""
          }
          a

let toString =
      \(t : Triple) ->
        let base = "${archToText t.arch}-${vendorToText t.vendor}-${osToText t.os}"
        in  merge
              { GNU = "${base}-gnu"
              , Musl = "${base}-musl"
              , MSVC = "${base}-msvc"
              , EABI = "${base}-eabi"
              , Android = "${base}-android"
              , NoABI = base
              }
              t.abi

-- Standard triples
let x86_64-linux-gnu =
      { arch = Arch.X86_64
      , vendor = Vendor.Unknown
      , os = OS.Linux
      , abi = ABI.GNU
      }

let x86_64-linux-musl =
      { arch = Arch.X86_64
      , vendor = Vendor.Unknown
      , os = OS.Linux
      , abi = ABI.Musl
      }

let aarch64-linux-gnu =
      { arch = Arch.AArch64
      , vendor = Vendor.Unknown
      , os = OS.Linux
      , abi = ABI.GNU
      }

let aarch64-linux-musl =
      { arch = Arch.AArch64
      , vendor = Vendor.Unknown
      , os = OS.Linux
      , abi = ABI.Musl
      }

let x86_64-darwin =
      { arch = Arch.X86_64
      , vendor = Vendor.Apple
      , os = OS.Darwin
      , abi = ABI.NoABI
      }

let aarch64-darwin =
      { arch = Arch.AArch64
      , vendor = Vendor.Apple
      , os = OS.Darwin
      , abi = ABI.NoABI
      }

let wasm32-wasi =
      { arch = Arch.WASM32
      , vendor = Vendor.Unknown
      , os = OS.WASI
      , abi = ABI.NoABI
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
