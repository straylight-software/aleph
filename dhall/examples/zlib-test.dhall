-- dhall/examples/zlib-test.dhall
--
-- Generate build script for zlib-test

let T = ../Target.dhall
let P = ../Platform.dhall
let B = ../Build.dhall

-- Mock toolchain (in reality, read from env/config)
let mockCxxToolchain : P.CxxToolchain =
    { clang = T.path "/nix/store/xxx/bin/clang"
    , clangxx = T.path "/nix/store/xxx/bin/clang++"
    , lld = T.path "/nix/store/xxx/bin/ld.lld"
    , ar = T.path "/nix/store/xxx/bin/llvm-ar"
    , resourceDir = T.path "/nix/store/xxx/lib/clang/22"
    , gccInclude = T.path "/nix/store/yyy/include/c++/15.2.0"
    , gccIncludeArch = T.path "/nix/store/yyy/include/c++/15.2.0/x86_64-unknown-linux-gnu"
    , glibcInclude = T.path "/nix/store/zzz/include"
    , gccLib = T.path "/nix/store/yyy/lib/gcc/x86_64-unknown-linux-gnu/15.2.0"
    , gccLibBase = T.path "/nix/store/yyy/lib"
    , glibcLib = T.path "/nix/store/zzz/lib"
    }

let zlibTest = B.buildCxx
    (T.name "zlib-test")
    [ T.path "main.cpp" ]
    T.defaults.cxx
    [ T.Dep.Nix (T.nixpkgs "zlib") ]
    mockCxxToolchain

in zlibTest.script
