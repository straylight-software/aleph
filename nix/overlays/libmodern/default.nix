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

  # Use the consolidated GHC 9.12 from aleph.script
  # This ensures all Haskell code uses the same GHC version
  ghc-with-deps = final.aleph.script.ghc;

  # ══════════════════════════════════════════════════════════════════════════════
  # HELPER SCRIPTS
  # ══════════════════════════════════════════════════════════════════════════════

  # combine-archive: Use Buck2-built version from aleph.script.compiled
  # Falls back to local build if Buck2 output not available yet
  combine-archive =
    final.aleph.script.compiled.combine-archive or (final.stdenv.mkDerivation {
      name = "combine-archive";
      src = ../../../src/tools/scripts;
      "dontUnpack" = true;

      "nativeBuildInputs" = [ ghc-with-deps ];

      "buildPhase" = ''
        runHook preBuild
        ghc -O2 -Wall -Wno-unused-imports \
          -hidir . -odir . \
          -i$src -o combine-archive $src/combine-archive.hs
        runHook postBuild
      '';

      "installPhase" = ''
        runHook preInstall
        mkdir -p $out/bin
        cp combine-archive $out/bin/
        runHook postInstall
      '';

      meta = {
        description = "Combine multiple static archives into a single .a file";
      };
    });

  # ══════════════════════════════════════════════════════════════════════════════
  # BUILDER HELPER
  # ══════════════════════════════════════════════════════════════════════════════

  # Standard CMake flags for all libmodern packages
  standard-cmake-flags = [
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
  #     cmake-flags = [ ... ];  # merged with standard flags
  #   }
  #
  mk-static-cpp =
    {
      pname,
      version,
      src,
      native-build-inputs ? [ ],
      build-inputs ? [ ],
      propagated-build-inputs ? [ ],
      cmake-flags ? [ ],
      post-install ? "",
      post-patch ? "",
      patches ? [ ],
      env ? [ ],
      meta ? { },
      ...
    }@args:
    let
      extra-args = builtins.removeAttrs args [
        "pname"
        "version"
        "src"
        "native-build-inputs"
        "build-inputs"
        "propagated-build-inputs"
        "cmake-flags"
        "post-install"
        "post-patch"
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
          ;

        "postPatch" = post-patch;

        "nativeBuildInputs" = [
          final.cmake
          final.pkg-config
        ]
        ++ native-build-inputs;

        "buildInputs" = build-inputs;
        "propagatedBuildInputs" = propagated-build-inputs;

        "cmakeFlags" = standard-cmake-flags ++ cmake-flags;

        "postInstall" = post-install;

        env = {
          "NIX_CFLAGS_COMPILE" = "-fPIC";
        }
        // env;

        meta = {
          platforms = lib.platforms.unix;
        }
        // meta;
      }
      // extra-args
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
    inherit mk-static-cpp standard-cmake-flags;

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
