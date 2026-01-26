let Build = /home/b7r6/src/straylight/aleph/src/armitage/dhall/Build.dhall
let Toolchain = /home/b7r6/src/straylight/aleph/src/armitage/dhall/Toolchain.dhall
let Resource = /home/b7r6/src/straylight/aleph/src/armitage/dhall/Resource.dhall

in  Build.cxx-binary
      { name = "test-fetch"
      , srcs = [ "test-fetch.cpp" ]
      , deps = [] : List Build.Dep
      , toolchain = Toolchain.presets.clang-18-glibc-dynamic
      , requires = Resource.pure
      }
