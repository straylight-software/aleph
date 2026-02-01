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
#   aleph.shortlist.enable = true;
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_:
{
  config,
  lib,
  flake-parts-lib,
  ...
}:
let
  # ──────────────────────────────────────────────────────────────────────────────
  # Lisp-case aliases for lib.* functions
  # ──────────────────────────────────────────────────────────────────────────────
  mk-enable-option = lib.mkEnableOption;
  mk-if = lib.mkIf;
  mk-per-system-option = flake-parts-lib.mkPerSystemOption;

  cfg = config.aleph.shortlist;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for aleph.shortlist
  # ════════════════════════════════════════════════════════════════════════════
  options.perSystem = mk-per-system-option (
    { lib, ... }:
    let
      mk-option' = lib.mkOption;
    in
    {
      options.aleph.shortlist = {
        libraries = mk-option' {
          type = lib.types.attrsOf lib.types.package;
          default = { };
          description = "Shortlist library packages";
        };
        shellHook = mk-option' {
          type = lib.types.lines;
          default = "";
          description = "Shell hook for shortlist setup";
        };
        buckconfig = mk-option' {
          type = lib.types.lines;
          default = "";
          description = "Buck2 config section for shortlist paths";
        };
        shortlist-file = mk-option' {
          type = lib.types.package;
          description = "Derivation containing the shortlist buckconfig section";
        };
      };
    }
  );

  options.aleph.shortlist = {
    enable = mk-enable-option "Hermetic C++ shortlist libraries";

    # Individual library toggles
    zlib-ng = mk-enable-option "zlib-ng compression library" // {
      default = true;
    };
    fmt = mk-enable-option "fmt formatting library" // {
      default = true;
    };
    catch2 = mk-enable-option "Catch2 testing framework" // {
      default = true;
    };
    spdlog = mk-enable-option "spdlog logging library" // {
      default = true;
    };
    mdspan = mk-enable-option "mdspan header-only library" // {
      default = true;
    };
    rapidjson = mk-enable-option "RapidJSON header-only library" // {
      default = true;
    };
    nlohmann-json = mk-enable-option "nlohmann-json header-only library" // {
      default = true;
    };
    libsodium = mk-enable-option "libsodium cryptography library" // {
      default = true;
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Implementation
  # ════════════════════════════════════════════════════════════════════════════
  config = mk-if cfg.enable {
    perSystem =
      {
        pkgs,
        lib,
        ...
      }:
      let
        optional-string' = lib.optionalString;

        # Use nixpkgs packages - they're already built
        # For true hermetic builds with LLVM 22, we'd override stdenv
        # Note: Many packages split headers into .dev output
        libraries = {
          inherit (pkgs) zlib-ng;
          inherit (pkgs) fmt;
          fmt-dev = pkgs.fmt.dev;
          catch2 = pkgs.catch2_3;
          catch2-dev = pkgs.catch2_3.dev or pkgs.catch2_3;
          inherit (pkgs) spdlog;
          spdlog-dev = pkgs.spdlog.dev or pkgs.spdlog;
          inherit (pkgs) mdspan;
          inherit (pkgs) rapidjson;
          nlohmann-json = pkgs.nlohmann_json;
          inherit (pkgs) libsodium;
          libsodium-dev = pkgs.libsodium.dev or pkgs.libsodium;
        };

        # Generate buckconfig section
        # Note: Use -dev outputs for include paths, regular for libs
        buckconfig-section = ''

          [shortlist]
          # Hermetic C++ libraries
          # Format: lib = /nix/store/..., lib_dev = /nix/store/...-dev (for headers)
          ${optional-string' cfg.zlib-ng "zlib_ng = ${libraries.zlib-ng}"}
          ${optional-string' cfg.fmt "fmt = ${libraries.fmt}"}
          ${optional-string' cfg.fmt "fmt_dev = ${libraries.fmt-dev}"}
          ${optional-string' cfg.catch2 "catch2 = ${libraries.catch2}"}
          ${optional-string' cfg.catch2 "catch2_dev = ${libraries.catch2-dev}"}
          ${optional-string' cfg.spdlog "spdlog = ${libraries.spdlog}"}
          ${optional-string' cfg.spdlog "spdlog_dev = ${libraries.spdlog-dev}"}
          ${optional-string' cfg.mdspan "mdspan = ${libraries.mdspan}"}
          ${optional-string' cfg.rapidjson "rapidjson = ${libraries.rapidjson}"}
          ${optional-string' cfg.nlohmann-json "nlohmann_json = ${libraries.nlohmann-json}"}
          ${optional-string' cfg.libsodium "libsodium = ${libraries.libsodium}"}
          ${optional-string' cfg.libsodium "libsodium_dev = ${libraries.libsodium-dev}"}
        '';

        # Shortlist section as a file in the store
        shortlist-file = pkgs.writeText "shortlist-section" buckconfig-section;

        # Shell hook to add shortlist section to .buckconfig.local
        shortlist-shell-hook = ''
          # Add shortlist section to .buckconfig.local
          if [ -f ".buckconfig.local" ]; then
            if ! grep -q "\\[shortlist\\]" .buckconfig.local 2>/dev/null; then
              cat ${shortlist-file} >> .buckconfig.local
              echo "Added [shortlist] section to .buckconfig.local"
            fi
          fi
        '';

      in
      {
        aleph.shortlist = {
          inherit libraries;
          buckconfig = buckconfig-section;
          inherit shortlist-file;
          shellHook = shortlist-shell-hook;
        };
      };
  };
}
