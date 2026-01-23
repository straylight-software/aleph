{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}

{- | Abseil C++ - Google's C++ common libraries

Abseil produces ~130 separate static libraries. We combine them into
a single libabseil.a with a clean pkg-config file using a postInstall
script that:

1. Parses pkg-config files to build a dependency graph
2. Topologically sorts libraries (Kahn's algorithm)
3. Combines them into libabseil.a using ar
4. Generates a single abseil.pc file

The Haskell version mirrors the Nix version at:
  nix/overlays/libmodern/abseil-cpp/default.nix
-}
module Aleph.Nix.Packages.AbseilCpp (
    abseilCpp,
    abseilCppVersion,
    combineArchiveScript,
) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax
import Data.Text (Text)
import qualified Data.Text as T

-- | Current version of abseil-cpp we package.
abseilCppVersion :: Text
abseilCppVersion = "20250127.1"

{- | Abseil C++ libraries (combined static archive).

Features:
  - All ~130 libraries combined into single libabseil.a
  - Clean pkg-config file (abseil.pc)
  - Inline namespace disabled for ABI stability
  - C++17, static, PIC
-}
abseilCpp :: Drv
abseilCpp =
    mkDerivation
        [ pname "abseil-cpp"
        , version abseilCppVersion
        , src $
            fetchFromGitHub
                [ owner "abseil"
                , repo "abseil-cpp"
                , rev abseilCppVersion
                , hash "sha256-QTywqQCkyGFpdbtDBvUwz9bGXxbJs/qoFKF6zYAZUmQ="
                ]
        , nativeBuildInputs ["cmake", "pkg-config"]
        , buildInputs ["gtest"]
        , -- Disable inline namespaces for ABI stability
          postPatch
            [ substitute
                "absl/base/options.h"
                [
                    ( "#define ABSL_OPTION_USE_INLINE_NAMESPACE 1"
                    , "#define ABSL_OPTION_USE_INLINE_NAMESPACE 0"
                    )
                ]
            ]
        , -- Typed CMake options
          cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , positionIndependentCode = Just True
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                , extraFlags =
                    [ ("ABSL_BUILD_TEST_HELPERS", "ON")
                    , ("ABSL_USE_EXTERNAL_GOOGLETEST", "ON")
                    ]
                }
        , -- Combine all libabsl_*.a into single libabseil.a
          postInstall
            [ run "bash" ["-c", combineArchiveScript]
            ]
        , description "Abseil C++ libraries (combined static archive)"
        , homepage "https://abseil.io"
        , license "apache-2.0"
        ]

{- | Script to combine Abseil's ~130 static libraries into one.

This is the Haskell-embedded version of combine-archive.sh.
It runs in $out/lib after cmake install.

The script:
1. Extracts private dependencies from pkg-config files
2. Builds a dependency graph from Requires fields
3. Topologically sorts using Kahn's algorithm
4. Combines archives using ar -M
5. Generates a clean abseil.pc file
-}
combineArchiveScript :: Text
combineArchiveScript =
    T.unlines
        [ "set -euo pipefail"
        , "cd $out/lib"
        , ""
        , "# Extract private dependencies from pkg-config files"
        , "PRIVATE_DEPS=\"pthread m rt dl\""
        , "for pc in pkgconfig/absl_*.pc; do"
        , "  if [ -f \"$pc\" ]; then"
        , "    grep -h \"Libs.private:\" \"$pc\" 2>/dev/null | sed 's/Libs.private://' | tr ' ' '\\n' | grep -E '^-l' | sed 's/^-l//' >>deps.tmp || true"
        , "  fi"
        , "done"
        , "if [ -f deps.tmp ]; then"
        , "  PRIVATE_DEPS=$(echo $PRIVATE_DEPS $(sort -u deps.tmp | tr '\\n' ' '))"
        , "  rm deps.tmp"
        , "fi"
        , ""
        , "# Build dependency graph and collect libraries"
        , "declare -A deps_graph"
        , "declare -A all_libs"
        , ""
        , "for pc in pkgconfig/absl_*.pc; do"
        , "  if [ -f \"$pc\" ]; then"
        , "    lib_name=$(basename \"$pc\" .pc)"
        , "    lib_file=\"lib${lib_name}.a\""
        , "    if [ -f \"$lib_file\" ]; then"
        , "      all_libs[\"$lib_name\"]=1"
        , "      deps=$(grep -E \"^Requires:|^Requires.private:\" \"$pc\" 2>/dev/null | \\"
        , "        sed 's/^[^:]*://' | tr ',' ' ' | grep -o 'absl_[^ ]*' | sort -u | tr '\\n' ' ' || echo \"\")"
        , "      deps_graph[\"$lib_name\"]=\"$deps\""
        , "    fi"
        , "  fi"
        , "done"
        , ""
        , "# Kahn's algorithm for topological sort"
        , "kahn_sort() {"
        , "  local -A in_degree"
        , "  local -a queue sorted"
        , "  for lib in \"${!all_libs[@]}\"; do in_degree[\"$lib\"]=0; done"
        , "  for lib in \"${!deps_graph[@]}\"; do"
        , "    for dep in ${deps_graph[\"$lib\"]}; do"
        , "      [[ -n ${all_libs[$dep]:-} ]] && ((in_degree[\"$dep\"]++))"
        , "    done"
        , "  done"
        , "  for lib in \"${!in_degree[@]}\"; do"
        , "    [[ ${in_degree[\"$lib\"]} -eq 0 ]] && queue+=(\"$lib\")"
        , "  done"
        , "  while [[ ${#queue[@]} -gt 0 ]]; do"
        , "    IFS=$'\\n' queue=($(sort <<<\"${queue[*]}\"))"
        , "    local current=\"${queue[0]}\""
        , "    queue=(\"${queue[@]:1}\")"
        , "    sorted+=(\"lib${current}.a\")"
        , "    for lib in \"${!deps_graph[@]}\"; do"
        , "      if [[ \" ${deps_graph[\"$lib\"]} \" == *\" $current \"* ]]; then"
        , "        ((in_degree[\"$lib\"]--)); [[ ${in_degree[\"$lib\"]} -eq 0 ]] && queue+=(\"$lib\")"
        , "      fi"
        , "    done"
        , "  done"
        , "  # Handle cycles by appending remaining"
        , "  for lib in \"${!all_libs[@]}\"; do"
        , "    [[ ${in_degree[\"$lib\"]:-0} -gt 0 ]] && sorted+=(\"lib${lib}.a\")"
        , "  done"
        , "  printf '%s\\n' \"${sorted[@]}\""
        , "}"
        , ""
        , "# Get sorted library list"
        , "if [[ ${#all_libs[@]} -gt 0 ]]; then"
        , "  LIBS=$(kahn_sort)"
        , "else"
        , "  LIBS=$(find . -name \"libabsl_*.a\" | sort)"
        , "fi"
        , ""
        , "# Combine archives using ar"
        , "echo \"CREATE libabseil.a\" >combine.ar"
        , "echo \"$LIBS\" | while read -r lib; do echo \"ADDLIB $lib\" >>combine.ar; done"
        , "echo \"SAVE\" >>combine.ar"
        , "echo \"END\" >>combine.ar"
        , "ar -M <combine.ar"
        , "rm combine.ar"
        , ""
        , "# Clean up individual archives"
        , "rm -f libabsl_*.a"
        , "find pkgconfig -name \"absl_*.pc\" -delete"
        , ""
        , "# Generate unified pkg-config file"
        , "cat >pkgconfig/abseil.pc <<EOF"
        , "prefix=$out"
        , "exec_prefix=\\${prefix}"
        , "libdir=\\${prefix}/lib"
        , "includedir=\\${prefix}/include"
        , ""
        , "Name: libabseil"
        , "Description: Abseil C++ libraries (libmodern combined archive)"
        , "Version: " <> abseilCppVersion
        , "URL: https://abseil.io/"
        , "Libs: -L\\${libdir} -labseil"
        , "Libs.private: -L\\${libdir}$(echo $PRIVATE_DEPS | xargs -n1 | sort -u | xargs printf ' -l%s')"
        , "Cflags: -I\\${includedir}"
        , "EOF"
        , ""
        , "echo \"// created libabseil.a ($(du -h libabseil.a | cut -f1))\""
        ]
