# toolchains/shortlist.bzl
#
# Prebuilt C++ libraries from the Nix shortlist module.
# Paths are read from .buckconfig.local [shortlist] section.
#
# Available targets (when shortlist_all() is called and shortlist is configured):
#   toolchains//:zlib-ng       - High-performance zlib replacement
#   toolchains//:fmt           - Modern C++ formatting library
#   toolchains//:catch2        - C++ testing framework
#   toolchains//:catch2-main   - Catch2 with main() provider
#   toolchains//:spdlog        - Fast C++ logging library
#   toolchains//:libsodium     - Modern cryptography library
#   toolchains//:mdspan        - C++23 multidimensional array view (header-only)
#   toolchains//:rapidjson     - Fast JSON parser/generator (header-only)
#   toolchains//:nlohmann-json - JSON for Modern C++ (header-only)

def _shortlist_path(name: str) -> str | None:
    """Get shortlist library path, returns None if not configured."""
    return read_root_config("shortlist", name, None)

def shortlist_all():
    """Define all shortlist library targets (only if [shortlist] section exists)."""

    # Check if shortlist is configured
    zlib_ng_path = _shortlist_path("zlib_ng")
    if zlib_ng_path == None:
        # Shortlist not configured, skip all definitions
        return

    # ════════════════════════════════════════════════════════════════════════════
    # zlib-ng - High-performance zlib replacement
    # ════════════════════════════════════════════════════════════════════════════
    native.prebuilt_cxx_library(
        name = "zlib-ng",
        exported_preprocessor_flags = ["-isystem", zlib_ng_path + "/include"],
        exported_linker_flags = [zlib_ng_path + "/lib/libz.a"],
        visibility = ["PUBLIC"],
    )

    # ════════════════════════════════════════════════════════════════════════════
    # fmt - Modern C++ formatting library
    # ════════════════════════════════════════════════════════════════════════════
    fmt_path = _shortlist_path("fmt")
    fmt_dev_path = _shortlist_path("fmt_dev")
    if fmt_path:
        native.prebuilt_cxx_library(
            name = "fmt",
            exported_preprocessor_flags = ["-isystem", (fmt_dev_path or fmt_path) + "/include"],
            # Link against shared lib (nixpkgs doesn't build static by default)
            exported_linker_flags = ["-L" + fmt_path + "/lib", "-lfmt", "-Wl,-rpath," + fmt_path + "/lib"],
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # catch2 - C++ testing framework
    # ════════════════════════════════════════════════════════════════════════════
    catch2_path = _shortlist_path("catch2")
    catch2_dev_path = _shortlist_path("catch2_dev")
    if catch2_path:
        native.prebuilt_cxx_library(
            name = "catch2",
            exported_preprocessor_flags = ["-isystem", (catch2_dev_path or catch2_path) + "/include"],
            exported_linker_flags = [catch2_path + "/lib/libCatch2.a"],
            visibility = ["PUBLIC"],
        )

        # Catch2 with main() - for tests that don't define their own main
        native.prebuilt_cxx_library(
            name = "catch2-main",
            exported_linker_flags = [catch2_path + "/lib/libCatch2Main.a"],
            exported_deps = [":catch2"],
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # spdlog - Fast C++ logging library
    # ════════════════════════════════════════════════════════════════════════════
    spdlog_path = _shortlist_path("spdlog")
    spdlog_dev_path = _shortlist_path("spdlog_dev")
    if spdlog_path:
        native.prebuilt_cxx_library(
            name = "spdlog",
            exported_preprocessor_flags = ["-isystem", (spdlog_dev_path or spdlog_path) + "/include"],
            exported_linker_flags = [spdlog_path + "/lib/libspdlog.a"],
            exported_deps = [":fmt"] if fmt_path else [],
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # libsodium - Modern cryptography library
    # ════════════════════════════════════════════════════════════════════════════
    libsodium_path = _shortlist_path("libsodium")
    libsodium_dev_path = _shortlist_path("libsodium_dev")
    if libsodium_path:
        native.prebuilt_cxx_library(
            name = "libsodium",
            exported_preprocessor_flags = ["-isystem", (libsodium_dev_path or libsodium_path) + "/include"],
            exported_linker_flags = [libsodium_path + "/lib/libsodium.a"],
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # mdspan - C++23 multidimensional array view (header-only)
    # ════════════════════════════════════════════════════════════════════════════
    mdspan_path = _shortlist_path("mdspan")
    if mdspan_path:
        native.prebuilt_cxx_library(
            name = "mdspan",
            exported_preprocessor_flags = ["-isystem", mdspan_path + "/include"],
            header_only = True,
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # rapidjson - Fast JSON parser/generator (header-only)
    # ════════════════════════════════════════════════════════════════════════════
    rapidjson_path = _shortlist_path("rapidjson")
    if rapidjson_path:
        native.prebuilt_cxx_library(
            name = "rapidjson",
            exported_preprocessor_flags = ["-isystem", rapidjson_path + "/include"],
            header_only = True,
            visibility = ["PUBLIC"],
        )

    # ════════════════════════════════════════════════════════════════════════════
    # nlohmann-json - JSON for Modern C++ (header-only)
    # ════════════════════════════════════════════════════════════════════════════
    nlohmann_json_path = _shortlist_path("nlohmann_json")
    if nlohmann_json_path:
        native.prebuilt_cxx_library(
            name = "nlohmann-json",
            exported_preprocessor_flags = ["-isystem", nlohmann_json_path + "/include"],
            header_only = True,
            visibility = ["PUBLIC"],
        )
