let Build = ../../../src/armitage/dhall/Build.dhall
let Toolchain = ../../../src/armitage/dhall/Toolchain.dhall
let Resource = ../../../src/armitage/dhall/Resource.dhall

-- Shell build that fetches - declares PURE (lying)
in  { name = "fetch-test"
    , srcs = Build.Src.Files [ "build.sh" ]
    , deps = [] : List Build.Dep
    , toolchain = Toolchain.presets.clang-18-glibc-dynamic
    , requires = Resource.pure  -- LIE: this build fetches
    }
