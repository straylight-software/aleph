# nix/build/spec.nix
#
# Package spec helpers for converting Dhall to Nix attrsets.
#
# Until we have dhall-to-nix evaluation at runtime, these are
# hand-written Nix equivalents of the Dhall packages.
#
{ lib }:

let
  # ════════════════════════════════════════════════════════════════════════════
  # SOURCE TYPES
  # ════════════════════════════════════════════════════════════════════════════

  github = owner: repo: rev: hash: {
    type = "GitHub";
    inherit
      owner
      repo
      rev
      hash
      ;
  };

  url = url': hash: {
    type = "Url";
    url = url';
    inherit hash;
  };

  local = path: {
    type = "Local";
    inherit path;
  };

  noSrc = {
    type = "None";
  };

  # ════════════════════════════════════════════════════════════════════════════
  # BUILD TYPES
  # ════════════════════════════════════════════════════════════════════════════

  cmakeBuild =
    {
      flags ? [ ],
      buildType ? "Release",
      linkage ? "Static",
      pic ? "Default",
      lto ? "Off",
    }:
    {
      type = "CMake";
      inherit
        flags
        buildType
        linkage
        pic
        lto
        ;
    };

  autotoolsBuild =
    {
      configureFlags ? [ ],
      makeFlags ? [ ],
    }:
    {
      type = "Autotools";
      inherit configureFlags makeFlags;
    };

  mesonBuild =
    {
      flags ? [ ],
      buildType ? "release",
    }:
    {
      type = "Meson";
      inherit flags buildType;
    };

  headerOnlyBuild =
    {
      includeDir ? "include",
    }:
    {
      type = "HeaderOnly";
      include = includeDir;
    };

  customBuild = builder: {
    type = "Custom";
    inherit builder;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PACKAGE DEFAULTS
  # ════════════════════════════════════════════════════════════════════════════

  defaults = {
    deps = [ ];
    target = null;
    checks = [ ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PACKAGE SPECS (Nix equivalents of packages-dhall/)
  # ════════════════════════════════════════════════════════════════════════════

  specs = {
    zlib-ng =
      host:
      defaults
      // {
        name = "zlib-ng";
        version = "2.2.4";
        src = github "zlib-ng" "zlib-ng" "2.2.4" "sha256-xJi0xFbHBd511z8H/ra22K2T1aSWcKjkxuyuwd/kvBg=";
        deps = [
          "cmake"
          "ninja"
        ];
        build = cmakeBuild {
          flags = [
            "-DZLIB_COMPAT=ON"
            "-DWITH_GTEST=OFF"
          ];
          linkage = "Static";
        };
        inherit host;
      };

    fmt =
      host:
      defaults
      // {
        name = "fmt";
        version = "11.0.2";
        src = github "fmtlib" "fmt" "11.0.2" "sha256-IKNt4xUoVi750zBti5iJJcCk3zivTt7nU12RIf8pM+0=";
        deps = [
          "cmake"
          "ninja"
        ];
        build = cmakeBuild {
          flags = [
            "-DFMT_TEST=OFF"
            "-DFMT_DOC=OFF"
          ];
          linkage = "Static";
        };
        inherit host;
      };

    mdspan =
      host:
      defaults
      // {
        name = "mdspan";
        version = "0.6.0";
        src = github "kokkos" "mdspan" "mdspan-0.6.0" "sha256-fvlK4xyVjWz7vYwCH/PN5KMEBxOoYi8gfNsWEyUI3po=";
        deps = [ ];
        build = headerOnlyBuild { includeDir = "include"; };
        inherit host;
      };

    rapidjson =
      host:
      defaults
      // {
        name = "rapidjson";
        version = "1.1.0";
        src = github "Tencent" "rapidjson" "v1.1.0" "sha256-SxUXV6K9lCD+wSjLR0X+cxuDjVQ0VEPWJ2YCLE2gKaE=";
        deps = [ ];
        build = headerOnlyBuild { includeDir = "include"; };
        inherit host;
      };

    nlohmann-json =
      host:
      defaults
      // {
        name = "nlohmann-json";
        version = "3.11.3";
        src = github "nlohmann" "json" "v3.11.3" "sha256-7F0Jon+1oWL7uqet5i1IgHX0fUw/+z0QwEcA3zs5xHg=";
        deps = [ ];
        build = headerOnlyBuild { includeDir = "include"; };
        inherit host;
      };

    catch2 =
      host:
      defaults
      // {
        name = "catch2";
        version = "3.7.1";
        src = github "catchorg" "Catch2" "v3.7.1" "sha256-xGPfXjk+oOnR7JqTrZd2pKJxalrlS8CMs7HWDClXii0=";
        deps = [
          "cmake"
          "ninja"
        ];
        build = cmakeBuild {
          flags = [
            "-DCATCH_BUILD_TESTING=OFF"
            "-DCATCH_INSTALL_DOCS=OFF"
          ];
          linkage = "Static";
        };
        inherit host;
      };

    spdlog =
      host:
      defaults
      // {
        name = "spdlog";
        version = "1.15.0";
        src = github "gabime" "spdlog" "v1.15.0" "sha256-sL2zHE1HLzFy6fUhLiKi0pz8/z+PjP0V8cV3VgJmC6c=";
        deps = [
          "cmake"
          "ninja"
          "fmt"
        ];
        build = cmakeBuild {
          flags = [
            "-DSPDLOG_BUILD_SHARED=OFF"
            "-DSPDLOG_BUILD_EXAMPLE=OFF"
            "-DSPDLOG_BUILD_TESTS=OFF"
            "-DSPDLOG_FMT_EXTERNAL=ON"
          ];
          linkage = "Static";
        };
        inherit host;
      };

    libsodium =
      host:
      defaults
      // {
        name = "libsodium";
        version = "1.0.20";
        src =
          github "jedisct1" "libsodium" "1.0.20-RELEASE"
            "sha256-OLQY0tXJbtKNz7v8o9GFzIXQvkzFT2nX3YGOXTq3fWI=";
        deps = [ ];
        build = autotoolsBuild {
          configureFlags = [
            "--disable-shared"
            "--enable-static"
          ];
        };
        inherit host;
      };
  };

in
{
  inherit
    # Source helpers
    github
    url
    local
    noSrc

    # Build helpers
    cmakeBuild
    autotoolsBuild
    mesonBuild
    headerOnlyBuild
    customBuild

    # Package defaults
    defaults

    # Pre-defined specs
    specs
    ;
}
