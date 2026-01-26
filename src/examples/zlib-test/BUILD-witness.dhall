let Build = ../../../src/armitage/dhall/Build.dhall
let Toolchain = ../../../src/armitage/dhall/Toolchain.dhall
let Resource = ../../../src/armitage/dhall/Resource.dhall

in  Build.cxx-binary
      { name = "test-witness"
      , srcs = [ "test-fetch.cpp" ]
      , deps = [] : List Build.Dep
      , toolchain = Toolchain.presets.clang-18-glibc-dynamic
      , requires = Resource.pure
      }
