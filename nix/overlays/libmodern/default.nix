# nix/overlays/libmodern/default.nix
#
# Modern C++ library overlay
#
# Provides:
#   - pkgs.libmodern.fmt
#   - pkgs.libmodern.abseil-cpp
#   - pkgs.libmodern.libsodium
#   - ... (more to come)
#
# Philosophy:
#   - Static libraries only (BUILD_SHARED_LIBS=OFF)
#   - C++17 minimum standard
#   - Position-independent code (-fPIC)
#   - RelWithDebInfo builds
#   - pkg-config as the interface contract
#
final: prev:
let
  inherit (prev) lib;

  # ══════════════════════════════════════════════════════════════════════════════
  # HASKELL DEPENDENCIES
  # ══════════════════════════════════════════════════════════════════════════════

  # GHC with dependencies for combine-archive script
  ghcWithDeps = final.haskellPackages.ghcWithPackages (
    ps: with ps; [
      shelly
      text
      aeson
      containers
      async
      foldl
    ]
  );

  # ══════════════════════════════════════════════════════════════════════════════
  # HELPER SCRIPTS
  # ══════════════════════════════════════════════════════════════════════════════

  # combine-archive: Combines multiple .a files into one using ar
  # Written in Haskell (no bash logic) per ℵ-006
  combine-archive = final.stdenv.mkDerivation {
    name = "combine-archive";
    src = ../../../src/tools/scripts;
    dontUnpack = true;

    nativeBuildInputs = [ ghcWithDeps ];

    buildPhase = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports \
        -hidir . -odir . \
        -i$src -o combine-archive $src/combine-archive.hs
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin
      cp combine-archive $out/bin/
      runHook postInstall
    '';

    meta = {
      description = "Combine multiple static archives into a single .a file";
    };
  };

  # ══════════════════════════════════════════════════════════════════════════════
  # BUILDER HELPER
  # ══════════════════════════════════════════════════════════════════════════════

  # Standard CMake flags for all libmodern packages
  standardCmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
    (lib.cmakeFeature "CMAKE_CXX_STANDARD" "17")
    (lib.cmakeBool "CMAKE_POSITION_INDEPENDENT_CODE" true)
    (lib.cmakeBool "BUILD_STATIC_LIBS" true)
    (lib.cmakeBool "BUILD_SHARED_LIBS" false)
  ];

  # Builder for static C++ libraries with standard flags
  #
  # Usage:
  #   mk-static-cpp {
  #     pname = "simdjson";
  #     version = "3.12.3";
  #     src = ...;
  #     cmakeFlags = [ ... ];  # merged with standard flags
  #   }
  #
  mk-static-cpp =
    {
      pname,
      version,
      src,
      nativeBuildInputs ? [ ],
      buildInputs ? [ ],
      propagatedBuildInputs ? [ ],
      cmakeFlags ? [ ],
      postInstall ? "",
      postPatch ? "",
      patches ? [ ],
      env ? { },
      meta ? { },
      ...
    }@args:
    let
      extraArgs = builtins.removeAttrs args [
        "pname"
        "version"
        "src"
        "nativeBuildInputs"
        "buildInputs"
        "propagatedBuildInputs"
        "cmakeFlags"
        "postInstall"
        "postPatch"
        "patches"
        "env"
        "meta"
      ];
    in
    final.stdenv.mkDerivation (
      {
        inherit
          pname
          version
          src
          patches
          postPatch
          ;

        nativeBuildInputs = [
          final.cmake
          final.pkg-config
        ]
        ++ nativeBuildInputs;

        inherit buildInputs propagatedBuildInputs;

        cmakeFlags = standardCmakeFlags ++ cmakeFlags;

        inherit postInstall;

        env = {
          NIX_CFLAGS_COMPILE = "-fPIC";
        }
        // env;

        meta = {
          platforms = lib.platforms.unix;
        }
        // meta;
      }
      // extraArgs
    );

  # ══════════════════════════════════════════════════════════════════════════════
  # PACKAGES
  # ══════════════════════════════════════════════════════════════════════════════

  fmt = import ./fmt.nix { inherit final lib mk-static-cpp; };
  libsodium = import ./libsodium.nix { inherit final lib; };
  abseil-cpp = import ./abseil-cpp {
    inherit
      final
      lib
      mk-static-cpp
      combine-archive
      ;
  };

in
{
  libmodern = {
    # Builder (exposed for custom packages)
    inherit mk-static-cpp standardCmakeFlags;

    # Helper scripts
    inherit combine-archive;

    # Packages
    inherit
      fmt
      libsodium
      abseil-cpp
      ;
  };
}
