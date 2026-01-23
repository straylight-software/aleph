# nix/modules/flake/shortlist.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // shortlist //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games, in early
#     graphics programs and military experimentation with cranial
#     jacks. On the Sony, a two-dimensional space war faded behind
#     a forest of mathematically generated ferns, demonstrating the
#     spatial possibilities of logarithmic spirals.
#
#                                                         — Neuromancer
#
# Hermetic C++ library builds with LLVM 22.
#
# The "shortlist" is the curated set of C++ libraries available for
# Buck2 builds. These libraries use absolute Nix store paths and are
# available as Buck2 prebuilt_cxx_library targets.
#
# Libraries:
#   - zlib-ng     : High-performance zlib replacement
#   - fmt         : Modern C++ formatting library
#   - catch2      : C++ testing framework
#   - spdlog      : Fast C++ logging library
#   - mdspan      : C++23 multidimensional array view (header-only)
#   - rapidjson   : Fast JSON parser/generator (header-only)
#   - nlohmann-json : JSON for Modern C++ (header-only)
#   - libsodium   : Modern cryptography library
#
# USAGE:
#
#   imports = [ aleph.modules.flake.shortlist ];
#
#   aleph-naught.shortlist.enable = true;
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  config,
  lib,
  flake-parts-lib,
  ...
}:
let
  inherit (flake-parts-lib) mkPerSystemOption;
  cfg = config.aleph-naught.shortlist;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for straylight.shortlist
  # ════════════════════════════════════════════════════════════════════════════
  options.perSystem = mkPerSystemOption (
    { lib, ... }:
    {
      options.straylight.shortlist = {
        libraries = lib.mkOption {
          type = lib.types.attrsOf lib.types.package;
          default = { };
          description = "Shortlist library packages";
        };
        shellHook = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Shell hook for shortlist setup";
        };
        buckconfig = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Buck2 config section for shortlist paths";
        };
      };
    }
  );

  options.aleph-naught.shortlist = {
    enable = lib.mkEnableOption "Hermetic C++ shortlist libraries";

    # Individual library toggles
    zlib-ng = lib.mkEnableOption "zlib-ng compression library" // {
      default = true;
    };
    fmt = lib.mkEnableOption "fmt formatting library" // {
      default = true;
    };
    catch2 = lib.mkEnableOption "Catch2 testing framework" // {
      default = true;
    };
    spdlog = lib.mkEnableOption "spdlog logging library" // {
      default = true;
    };
    mdspan = lib.mkEnableOption "mdspan header-only library" // {
      default = true;
    };
    rapidjson = lib.mkEnableOption "RapidJSON header-only library" // {
      default = true;
    };
    nlohmann-json = lib.mkEnableOption "nlohmann-json header-only library" // {
      default = true;
    };
    libsodium = lib.mkEnableOption "libsodium cryptography library" // {
      default = true;
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Implementation
  # ════════════════════════════════════════════════════════════════════════════
  config = lib.mkIf cfg.enable {
    perSystem =
      {
        pkgs,
        lib,
        ...
      }:
      let
        # Use nixpkgs packages - they're already built
        # For true hermetic builds with LLVM 22, we'd override stdenv
        # Note: Many packages split headers into .dev output
        libraries = {
          zlib-ng = pkgs.zlib-ng;
          fmt = pkgs.fmt;
          fmt-dev = pkgs.fmt.dev;
          catch2 = pkgs.catch2_3;
          catch2-dev = pkgs.catch2_3.dev or pkgs.catch2_3;
          spdlog = pkgs.spdlog;
          spdlog-dev = pkgs.spdlog.dev or pkgs.spdlog;
          mdspan = pkgs.mdspan;
          rapidjson = pkgs.rapidjson;
          nlohmann-json = pkgs.nlohmann_json;
          libsodium = pkgs.libsodium;
          libsodium-dev = pkgs.libsodium.dev or pkgs.libsodium;
        };

        # Generate buckconfig section
        # Note: Use -dev outputs for include paths, regular for libs
        buckconfigSection = ''

          [shortlist]
          # Hermetic C++ libraries
          # Format: lib = /nix/store/..., lib_dev = /nix/store/...-dev (for headers)
          ${lib.optionalString cfg.zlib-ng "zlib_ng = ${libraries.zlib-ng}"}
          ${lib.optionalString cfg.fmt "fmt = ${libraries.fmt}"}
          ${lib.optionalString cfg.fmt "fmt_dev = ${libraries.fmt-dev}"}
          ${lib.optionalString cfg.catch2 "catch2 = ${libraries.catch2}"}
          ${lib.optionalString cfg.catch2 "catch2_dev = ${libraries.catch2-dev}"}
          ${lib.optionalString cfg.spdlog "spdlog = ${libraries.spdlog}"}
          ${lib.optionalString cfg.spdlog "spdlog_dev = ${libraries.spdlog-dev}"}
          ${lib.optionalString cfg.mdspan "mdspan = ${libraries.mdspan}"}
          ${lib.optionalString cfg.rapidjson "rapidjson = ${libraries.rapidjson}"}
          ${lib.optionalString cfg.nlohmann-json "nlohmann_json = ${libraries.nlohmann-json}"}
          ${lib.optionalString cfg.libsodium "libsodium = ${libraries.libsodium}"}
          ${lib.optionalString cfg.libsodium "libsodium_dev = ${libraries.libsodium-dev}"}
        '';

        # Shell hook to add shortlist section to .buckconfig.local
        shortlistShellHook = ''
                    # Add shortlist section to .buckconfig.local
                    if [ -f ".buckconfig.local" ]; then
                      if ! grep -q "\\[shortlist\\]" .buckconfig.local 2>/dev/null; then
                        cat >> .buckconfig.local << 'SHORTLIST_EOF'
          ${buckconfigSection}
          SHORTLIST_EOF
                        echo "Added [shortlist] section to .buckconfig.local"
                      fi
                    fi
        '';

      in
      {
        straylight.shortlist = {
          inherit libraries;
          buckconfig = buckconfigSection;
          shellHook = shortlistShellHook;
        };
      };
  };
}
