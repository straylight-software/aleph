-- nix/modules/flake/build/scripts/buckconfig-haskell.dhall
--
-- Haskell toolchain configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let ghc : Text = env:GHC as Text
let ghc_pkg : Text = env:GHC_PKG as Text
let haddock : Text = env:HADDOCK as Text
let ghc_version : Text = env:GHC_VERSION as Text
let ghc_lib_dir : Text = env:GHC_LIB_DIR as Text
let global_package_db : Text = env:GLOBAL_PACKAGE_DB as Text

in ''

[haskell]
ghc = ${ghc}
ghc_pkg = ${ghc_pkg}
haddock = ${haddock}
ghc_version = ${ghc_version}
ghc_lib_dir = ${ghc_lib_dir}
global_package_db = ${global_package_db}
''
