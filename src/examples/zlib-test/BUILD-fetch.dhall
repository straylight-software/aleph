let Build = ../../../src/armitage/dhall/Build.dhall
let Toolchain = ../../../src/armitage/dhall/Toolchain.dhall
let Resource = ../../../src/armitage/dhall/Resource.dhall

-- This target declares it needs network (honest)
in  Build.cxx-binary
      { name = "test-fetch-real"
      , srcs = [ "test-fetch-real.cpp" ]
      , deps = [] : List Build.Dep
      , toolchain = Toolchain.presets.clang-18-glibc-dynamic
      , requires = Resource.network  -- declares network access
      }
