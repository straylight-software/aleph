--| C/C++ Rules
--|
--| cxx_library, cxx_binary, cxx_test

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- C++ Standards
-- =============================================================================

let CxxStandard = < Cxx11 | Cxx14 | Cxx17 | Cxx20 | Cxx23 >

let CStandard = < C99 | C11 | C17 | C23 >

-- =============================================================================
-- cxx_library
-- =============================================================================

let CxxLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , hdrs : List Text
      , deps : List Types.Dep
      , exported_deps : List Types.Dep
      , includes : List Text
      , defines : List { name : Text, value : Optional Text }
      , cxx_std : CxxStandard
      , c_std : CStandard
      , copts : List Types.Flag
      , linkopts : List Types.Flag
      , pic : Bool
      , toolchain : Optional Toolchain.Toolchain
      }

let cxx_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(hdrs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , hdrs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , exported_deps = [] : List Types.Dep
        , includes = [] : List Text
        , defines = [] : List { name : Text, value : Optional Text }
        , cxx_std = CxxStandard.Cxx17
        , c_std = CStandard.C17
        , copts = [] : List Types.Flag
        , linkopts = [] : List Types.Flag
        , pic = True
        , toolchain = None Toolchain.Toolchain
        } : CxxLibrary

-- =============================================================================
-- cxx_binary
-- =============================================================================

let CxxBinary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , includes : List Text
      , defines : List { name : Text, value : Optional Text }
      , cxx_std : CxxStandard
      , copts : List Types.Flag
      , linkopts : List Types.Flag
      , link_style : < Static | Shared >
      , toolchain : Optional Toolchain.Toolchain
      }

let cxx_binary =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , includes = [] : List Text
        , defines = [] : List { name : Text, value : Optional Text }
        , cxx_std = CxxStandard.Cxx17
        , copts = [] : List Types.Flag
        , linkopts = [] : List Types.Flag
        , link_style = < Static | Shared >.Static
        , toolchain = None Toolchain.Toolchain
        } : CxxBinary

-- =============================================================================
-- cxx_test
-- =============================================================================

let CxxTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , includes : List Text
      , cxx_std : CxxStandard
      , copts : List Types.Flag
      , toolchain : Optional Toolchain.Toolchain
      }

let cxx_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , includes = [] : List Text
        , cxx_std = CxxStandard.Cxx17
        , copts = [] : List Types.Flag
        , toolchain = None Toolchain.Toolchain
        } : CxxTest

-- =============================================================================
-- Exports
-- =============================================================================

in  { CxxStandard
    , CStandard
    , CxxLibrary
    , cxx_library
    , CxxBinary
    , cxx_binary
    , CxxTest
    , cxx_test
    }
