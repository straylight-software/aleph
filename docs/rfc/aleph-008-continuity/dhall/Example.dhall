--| Example BUILD.dhall
--|
--| A real polyglot cross-compiling build configuration.
--| No globs. No strings. Real types.
let DICE = ./package.dhall

let rustc_artifact = { hash.sha256 = "def456...", name = "rustc-1.80.0" }

let orin_sysroot = { hash.sha256 = "deadbeef...", name = "jetpack-6-sysroot" }

let native_rust =
      DICE.nativeToolchain
        (DICE.rustc "1.80.0" rustc_artifact)
        [ DICE.Flag.OptLevel DICE.OptLevel.O2, DICE.Flag.LTO DICE.LTOMode.Thin ]

let orin_rust =
      DICE.crossToolchain
        (DICE.rustc "1.80.0" rustc_artifact)
        DICE.orin
        orin_sysroot
        [ DICE.Flag.OptLevel DICE.OptLevel.O2
        , DICE.Flag.TargetCpu DICE.Cpu.cortex_a78ae
        ]

let wasm_rust =
      DICE.crossToolchain
        (DICE.rustc "1.80.0" rustc_artifact)
        DICE.wasm32_wasi
        { hash.sha256 = "wasi...", name = "wasi-sysroot" }
        [ DICE.Flag.OptLevel DICE.OptLevel.Oz ]

let sha256_core = DICE.lean_library "sha256" [ "core/sha256.lean" ] []

let r2_backend =
      DICE.lean_library
        "r2-backend"
        [ "core/r2_backend.lean" ]
        [ "//core:sha256" ]

let git_odb =
      DICE.lean_library
        "git-odb"
        [ "core/git_odb.lean" ]
        [ "//core:r2-backend" ]

let builtins_wasm = DICE.wasm_module "builtins" "//core:git-odb"

let straylight_core =
      DICE.rust_library
        "straylight-core"
        [ "src/core/lib.rs"
        , "src/core/store.rs"
        , "src/core/artifact.rs"
        , "src/core/namespace.rs"
        , "src/core/vm.rs"
        ]
        [ "//vendor:wasmtime"
        , "//vendor:git2"
        , "//vendor:aws-sdk-s3"
        , "//vendor:ed25519-dalek"
        ]

let straylight_cli =
      DICE.rust_binary
        "straylight"
        [ "src/main.rs"
        , "src/cli/mod.rs"
        , "src/cli/build.rs"
        , "src/cli/run.rs"
        ]
        [ "//src:straylight-core" ]

let builtins_ffi =
      DICE.c_library
        "builtins-ffi"
        [ "src/ffi/builtins.c" ]
        [ "src/ffi/builtins.h" ]
        [ "//core:builtins-wasm" ]

in  { sha256_core
    , r2_backend
    , git_odb
    , builtins_wasm
    , straylight_core
    , straylight_cli
    , builtins_ffi
      -- Toolchains for reference
    , toolchains = { native = native_rust, orin = orin_rust, wasm = wasm_rust }
    }
