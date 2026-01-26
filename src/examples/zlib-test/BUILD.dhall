-- BUILD.dhall - zlib test target
-- 
-- Demonstrates building against nixpkgs dependencies.
-- DICE will resolve nixpkgs#zlib, get the store path,
-- and wire up -I and -L flags automatically.

let Build = ../../../src/armitage/dhall/Build.dhall
let Toolchain = ../../../src/armitage/dhall/Toolchain.dhall
let Resource = ../../../src/armitage/dhall/Resource.dhall

in  Build.cxx-binary
      { name = "zlib-test"
      , srcs = [ "main.cpp" ]
      , deps = [ Build.dep.nixpkgs "zlib" ]
      , toolchain = Toolchain.presets.clang-18-glibc-dynamic
      , requires = Resource.pure
      }
