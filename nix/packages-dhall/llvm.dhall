-- llvm: compiler infrastructure

let Drv = ../Drv/Prelude.dhall

let llvmTarget =
      \(arch : Drv.Arch) ->
        merge
          { x86_64 = "X86"
          , aarch64 = "AArch64"
          , armv7 = "ARM"
          , riscv64 = "RISCV"
          , wasm32 = "WebAssembly"
          , powerpc64le = "PowerPC"
          }
          arch

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "llvm"
          , version = "19.1.0"
          , src =
              Drv.github
                "llvm"
                "llvm-project"
                "llvmorg-19.1.0"
                "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX="
          , deps = [ "cmake", "ninja", "python3", "zlib-ng" ]
          , build =
              Drv.cmake
                (   Drv.CMake.defaults
                  //  { flags =
                          [ "-DLLVM_ENABLE_PROJECTS=clang;lld;clang-tools-extra"
                          , "-DLLVM_TARGETS_TO_BUILD=${llvmTarget host.arch}"
                          , "-DLLVM_ENABLE_RTTI=ON"
                          , "-DLLVM_ENABLE_ASSERTIONS=OFF"
                          , "-DLLVM_ENABLE_ZLIB=FORCE_ON"
                          , "-DLLVM_BUILD_LLVM_DYLIB=OFF"
                          , "-DLLVM_LINK_LLVM_DYLIB=OFF"
                          ]
                      , linkage = Drv.Linkage.Static
                      , lto = Drv.LTO.Thin
                      }
                )
          , host
          }
