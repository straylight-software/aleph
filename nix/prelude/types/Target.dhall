--| Target Triple Types
--|
--| Real triples, real architectures. No strings.

let Arch =
      < x86-64
      | aarch64
      | wasm32
      >

let OS =
      < linux
      | darwin
      | wasi
      >

let ABI =
      < gnu
      | musl
      | none
      >

let Cpu =
      < generic
      | native
      -- x86-64
      | znver3           -- AMD Zen 3
      | znver4           -- AMD Zen 4
      | sapphirerapids   -- Intel Sapphire Rapids
      -- aarch64 datacenter (SBSA)
      | neoverse-v2      -- Grace (GH200, GB200, DGX Spark)
      -- aarch64 embedded (Jetson)
      | cortex-a78ae     -- Orin (AGX/NX/Nano), Thor
      -- aarch64 consumer
      | apple-m1
      | apple-m2
      | apple-m3
      >

let Gpu =
      < none
      -- Ada Lovelace
      | sm-89            -- RTX 40xx
      -- Hopper
      | sm-90            -- H100 (datacenter)
      | sm-90a           -- H100 SXM (with async features)
      -- Orin
      | sm-87            -- Jetson Orin
      -- Blackwell
      | sm-100           -- B100, B200
      | sm-120           -- B200 (full features), RTX 50xx
      >

let Triple =
      { arch : Arch
      , os : OS
      , abi : ABI
      , cpu : Cpu
      }

let to-string
    : Triple -> Text
    = \(t : Triple) ->
        let arch =
              merge
                { x86-64 = "x86_64"
                , aarch64 = "aarch64"
                , wasm32 = "wasm32"
                }
                t.arch

        let os =
              merge
                { linux = "unknown-linux"
                , darwin = "apple-darwin"
                , wasi = "wasi"
                }
                t.os

        let abi =
              merge
                { gnu = "-gnu"
                , musl = "-musl"
                , none = ""
                }
                t.abi

        in  "${arch}-${os}${abi}"

--------------------------------------------------------------------------------
-- x86-64 targets
--------------------------------------------------------------------------------

let x86-64-linux =
      { arch = Arch.x86-64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.generic
      }

let x86-64-linux-musl =
      { arch = Arch.x86-64
      , os = OS.linux
      , abi = ABI.musl
      , cpu = Cpu.generic
      }

--------------------------------------------------------------------------------
-- aarch64 datacenter (SBSA / Grace)
--------------------------------------------------------------------------------

let grace =
      { arch = Arch.aarch64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.neoverse-v2
      }

--------------------------------------------------------------------------------
-- aarch64 embedded (Jetson)
--------------------------------------------------------------------------------

let orin =
      { arch = Arch.aarch64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.cortex-a78ae
      }

let thor =
      { arch = Arch.aarch64
      , os = OS.linux
      , abi = ABI.gnu
      , cpu = Cpu.cortex-a78ae
      }

--------------------------------------------------------------------------------
-- aarch64 consumer (Apple Silicon)
--------------------------------------------------------------------------------

let aarch64-darwin =
      { arch = Arch.aarch64
      , os = OS.darwin
      , abi = ABI.none
      , cpu = Cpu.apple-m1
      }

--------------------------------------------------------------------------------
-- wasm
--------------------------------------------------------------------------------

let wasm32-wasi =
      { arch = Arch.wasm32
      , os = OS.wasi
      , abi = ABI.none
      , cpu = Cpu.generic
      }

in  { Arch
    , OS
    , ABI
    , Cpu
    , Gpu
    , Triple
    , to-string
    -- x86-64
    , x86-64-linux
    , x86-64-linux-musl
    -- aarch64 datacenter
    , grace
    -- aarch64 embedded
    , orin
    , thor
    -- aarch64 consumer
    , aarch64-darwin
    -- wasm
    , wasm32-wasi
    }
