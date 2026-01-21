-- Drv/Buck.dhall
-- Emit Starlark from Package

let Prelude =
      https://prelude.dhall-lang.org/v23.1.0/package.dhall
        sha256:931cbfae9d746c4611b07633ab1e547637ab4ba138b16bf65ef1b9ad66a60b7f

let Package = ./Package.dhall
let Build = ./Build.dhall
let Flags = ./Flags.dhall

let quote = \(x : Text) -> "\"${x}\""

let renderList =
      \(xs : List Text) ->
        "[${Prelude.Text.concatSep ", " (Prelude.List.map Text Text quote xs)}]"

let linkageFlag =
      \(l : Flags.Linkage) ->
        merge
          { Static = [ "-DBUILD_SHARED_LIBS=OFF" ]
          , Shared = [ "-DBUILD_SHARED_LIBS=ON" ]
          , Both = [] : List Text
          }
          l

let linkageAutotools =
      \(l : Flags.Linkage) ->
        merge
          { Static = [ "--disable-shared", "--enable-static" ]
          , Shared = [ "--enable-shared", "--disable-static" ]
          , Both = [ "--enable-shared", "--enable-static" ]
          }
          l

let toStarlark =
      \(pkg : Package.Package) ->
        merge
          { CMake =
              \(c : Build.CMake.CMake) ->
                let allFlags = c.flags # linkageFlag c.linkage
                in  ''
                    cmake_library(
                        name = "${pkg.name}",
                        version = "${pkg.version}",
                        deps = ${renderList pkg.deps},
                        cmake_flags = ${renderList allFlags},
                    )
                    ''
          , Autotools =
              \(a : Build.Autotools.Autotools) ->
                let allFlags = a.configureFlags # linkageAutotools a.linkage
                in  ''
                    autotools_library(
                        name = "${pkg.name}",
                        version = "${pkg.version}",
                        deps = ${renderList pkg.deps},
                        configure_flags = ${renderList allFlags},
                    )
                    ''
          , HeaderOnly =
              \(h : { include : Text }) ->
                ''
                cxx_library(
                    name = "${pkg.name}",
                    version = "${pkg.version}",
                    exported_headers = glob(["${h.include}/**/*.h", "${h.include}/**/*.hpp"]),
                    header_only = True,
                )
                ''
          , Meson =
              \(m : Build.Meson.Meson) ->
                ''
                meson_library(
                    name = "${pkg.name}",
                    version = "${pkg.version}",
                    deps = ${renderList pkg.deps},
                )
                ''
          , Custom =
              \(c : { builder : Text }) ->
                ''
                genrule(
                    name = "${pkg.name}",
                    version = "${pkg.version}",
                    cmd = "aleph-build ${c.builder}",
                    deps = ${renderList pkg.deps},
                )
                ''
          }
          pkg.build

in  { toStarlark, renderList, quote }
