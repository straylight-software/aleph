--| Target Triple Types
--|
--| Real triples, real architectures. No strings.

let Arch =
      < x86_64
      | aarch64
      | wasm32
      | riscv64
      >

let OS =
      < linux
      | darwin
      | wasi
      | none
      >

let ABI =
      < gnu
      | musl
      | eabi
      | unknown
      >

let Cpu =
      < generic
      | native
      -- x86_64
      | znver3        -- AMD Zen 3
      | znver4        -- AMD Zen 4
      | skylake
      | icelake_server
      | sapphirerapids
      -- aarch64
      | cortex_a78ae  -- Jetson Orin
      | neoverse_n1   -- Graviton 2
      | neoverse_v1   -- Graviton 3
      | apple_m1
      | apple_m2
      | apple_m3
      >

let Triple =
      { arch : Arch
      , os : OS
      , abi : ABI
      , cpu : Cpu
      }

let tripleToString
    : Triple -> Text
    = \(t : Triple) ->
        let archStr =
              merge
                { x86_64 = "x86_64"
                , aarch64 = "aarch64"
                , wasm32 = "wasm32"
                , riscv64 = "riscv64"
                }
                t.arch

        let osStr =
              merge
                { linux = "unknown-linux"
                , darwin = "apple-darwin"
                , wasi = "wasi"
                , none = "unknown-none"
                }
                t.os

        let abiStr =
              merge
                { gnu = "gnu"
                , musl = "musl"
                , eabi = "eabi"
                , unknown = ""
                }
                t.abi

        in  "${archStr}-${osStr}${if abiStr == "" then "" else "-${abiStr}"}"

-- Common targets
let x86_64_linux =
      { arch = Arch.x86_64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.generic
      }

let aarch64_linux =
      { arch = Arch.aarch64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.generic
      }

let wasm32_wasi =
      { arch = Arch.wasm32
      , os = OS.wasi
      , abi = ABI.unknown
      , cpu = Cpu.generic
      }

let orin =
      { arch = Arch.aarch64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.cortex_a78ae
      }

in  { Arch
    , OS
    , ABI
    , Cpu
    , Triple
    , tripleToString
    , x86_64_linux
    , aarch64_linux
    , wasm32_wasi
    , orin
    }
