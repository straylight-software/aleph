-- dhall/Build.dhall
--
-- Target -> BuildScript
--
-- Generates shell scripts that compile targets.

let T = ./Target.dhall
let P = ./Platform.dhall

let Prelude = https://prelude.dhall-lang.org/v23.0.0/package.dhall

let map = Prelude.List.map
let filter = Prelude.List.filter
let length = Prelude.List.length
let isZero = Prelude.Natural.isZero
let concatSep = Prelude.Text.concatSep

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Build script output
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let BuildScript = { script : Text }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Helpers
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let pathToText : T.Path -> Text = \(p : T.Path) -> p._path
let nameToText : T.Name -> Text = \(n : T.Name) -> n._name

let spaceSep : List Text -> Text = concatSep " "

let cxxStdFlag : T.CxxStd -> Text = \(s : T.CxxStd) ->
    merge
        { C23 = "-std=c23"
        , Cxx17 = "-std=c++17"
        , Cxx20 = "-std=c++20"
        , Cxx23 = "-std=c++23"
        }
        s

let smArchFlag : T.SmArch -> Text = \(a : T.SmArch) ->
    merge
        { SM_100 = "--cuda-gpu-arch=sm_100"
        , SM_120 = "--cuda-gpu-arch=sm_120"
        }
        a

let rustEditionFlag : T.RustEdition -> Text = \(e : T.RustEdition) ->
    merge
        { E2015 = "--edition=2015"
        , E2018 = "--edition=2018"
        , E2021 = "--edition=2021"
        , E2024 = "--edition=2024"
        }
        e

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- C++ build
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let buildCxx
    : T.Name -> List T.Path -> T.CxxOpts -> List T.Dep -> P.CxxToolchain -> BuildScript
    = \(name : T.Name) ->
      \(srcs : List T.Path) ->
      \(opts : T.CxxOpts) ->
      \(deps : List T.Dep) ->
      \(tc : P.CxxToolchain) ->
        let srcFiles = spaceSep (map T.Path Text pathToText srcs)
        let std = cxxStdFlag opts.std
        let cflags = spaceSep opts.cflags
        let ldflags = spaceSep opts.ldflags
        let outName = nameToText name

        -- Nix deps handling
        let nixRefs = spaceSep (map T.Dep Text
            (\(d : T.Dep) -> merge
                { Local = \(_ : T.Label) -> ""
                , Nix = \(f : T.FlakeRef) -> "${f.flake}#${f.attr}"
                }
                d)
            deps)

        let hasNixDeps = isZero (length T.Dep
            (filter T.Dep
                (\(d : T.Dep) -> merge { Local = \(_ : T.Label) -> False, Nix = \(_ : T.FlakeRef) -> True } d)
                deps))

        let nixResolve = if hasNixDeps then "" else "NIX_FLAGS=$(\"$ANALYZER\" resolve ${nixRefs})"

        in { script =
''
#!/bin/bash
set -e

${nixResolve}

${pathToText tc.clangxx} \
    ${std} \
    -resource-dir ${pathToText tc.resourceDir} \
    -isystem ${pathToText tc.gccInclude} \
    -isystem ${pathToText tc.gccIncludeArch} \
    -isystem ${pathToText tc.glibcInclude} \
    ${cflags} \
    $NIX_FLAGS \
    ${srcFiles} \
    -o "$OUT" \
    -B ${pathToText tc.gccLib} -L ${pathToText tc.gccLib} \
    -L ${pathToText tc.gccLibBase} \
    -B ${pathToText tc.glibcLib} -L ${pathToText tc.glibcLib} \
    ${ldflags}
''
        }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- NV build (CUDA via clang)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let buildNv
    : T.Name -> List T.Path -> T.NvOpts -> List T.Dep -> P.NvToolchain -> BuildScript
    = \(name : T.Name) ->
      \(srcs : List T.Path) ->
      \(opts : T.NvOpts) ->
      \(deps : List T.Dep) ->
      \(tc : P.NvToolchain) ->
        let srcFiles = spaceSep (map T.Path Text pathToText srcs)
        let std = cxxStdFlag opts.cxxOpts.std
        let arch = smArchFlag opts.arch
        let cflags = spaceSep opts.cxxOpts.cflags
        let outName = nameToText name
        let cxxTc = tc.cxx

        in { script =
''
#!/bin/bash
set -e

${pathToText cxxTc.clangxx} \
    -x cuda \
    ${arch} \
    ${std} \
    -resource-dir ${pathToText cxxTc.resourceDir} \
    -isystem ${pathToText tc.nvidiaSdkInclude} \
    -isystem ${pathToText cxxTc.gccInclude} \
    -isystem ${pathToText cxxTc.gccIncludeArch} \
    -isystem ${pathToText cxxTc.glibcInclude} \
    ${cflags} \
    ${srcFiles} \
    -o "$OUT" \
    -L ${pathToText tc.nvidiaSdkLib} \
    -lcudart \
    -B ${pathToText cxxTc.gccLib} -L ${pathToText cxxTc.gccLib} \
    -L ${pathToText cxxTc.gccLibBase} \
    -B ${pathToText cxxTc.glibcLib} -L ${pathToText cxxTc.glibcLib}
''
        }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Haskell build
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let buildHaskell
    : T.Name -> List T.Path -> T.HaskellOpts -> List T.Dep -> P.HaskellToolchain -> BuildScript
    = \(name : T.Name) ->
      \(srcs : List T.Path) ->
      \(opts : T.HaskellOpts) ->
      \(deps : List T.Dep) ->
      \(tc : P.HaskellToolchain) ->
        let srcFiles = spaceSep (map T.Path Text pathToText srcs)
        let outName = nameToText name
        let pkgFlags = spaceSep (map Text Text (\(p : Text) -> "-package ${p}") opts.packages)
        let ghcOpts = spaceSep opts.ghcOptions
        let includePaths = spaceSep (map T.Path Text (\(p : T.Path) -> "-i${pathToText p}") opts.includePaths)

        in { script =
''
#!/bin/bash
set -e

${pathToText tc.ghc} \
    -O2 \
    ${ghcOpts} \
    ${pkgFlags} \
    ${includePaths} \
    ${srcFiles} \
    -o "$OUT"
''
        }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Rust build
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let buildRust
    : T.Name -> List T.Path -> T.RustOpts -> List T.Dep -> P.RustToolchain -> BuildScript
    = \(name : T.Name) ->
      \(srcs : List T.Path) ->
      \(opts : T.RustOpts) ->
      \(deps : List T.Dep) ->
      \(tc : P.RustToolchain) ->
        let srcFiles = spaceSep (map T.Path Text pathToText srcs)
        let outName = nameToText name
        let edition = rustEditionFlag opts.edition

        in { script =
''
#!/bin/bash
set -e

${pathToText tc.rustc} \
    ${edition} \
    -O \
    ${srcFiles} \
    -o "$OUT"
''
        }

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Lean build
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let buildLean
    : T.Name -> List T.Path -> T.LeanOpts -> List T.Dep -> P.LeanToolchain -> BuildScript
    = \(name : T.Name) ->
      \(srcs : List T.Path) ->
      \(opts : T.LeanOpts) ->
      \(deps : List T.Dep) ->
      \(tc : P.LeanToolchain) ->
        let srcFiles = spaceSep (map T.Path Text pathToText srcs)
        let outName = nameToText name

        in { script =
''
#!/bin/bash
set -e

${pathToText tc.lean} \
    ${srcFiles} \
    -o "$OUT"
''
        }

in  { BuildScript
    , buildCxx
    , buildNv
    , buildHaskell
    , buildRust
    , buildLean
    }
