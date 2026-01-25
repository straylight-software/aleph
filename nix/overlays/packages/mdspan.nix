# nix/packages/mdspan.nix — Kokkos reference implementation of std::mdspan
#
# P0009 mdspan - multidimensional array view for C++23
# GCC 15 doesn't ship it yet, so we use the Kokkos reference impl.
# Patched to inject into std:: namespace (not std::experimental::)
#
{
  lib,
  stdenv,
  pkgs,
  cmake,
}:
let
  # Import prelude for translate-attrs
  translations = import ../../prelude/translations.nix { inherit lib; };
  inherit (translations) translate-attrs;

  # External API alias
  fetch-from-github = pkgs.fetchFromGitHub;

  mdspan-shim = ./mdspan-shim.hpp;
in
stdenv.mkDerivation (
  final-attrs:
  translate-attrs {
    pname = "mdspan";
    version = "0.6.0";

    src = fetch-from-github {
      owner = "kokkos";
      repo = "mdspan";
      rev = "mdspan-${final-attrs.version}";
      hash = "sha256-bwE+NO/n9XsWOp3GjgLHz3s0JR0CzNDernfLHVqU9Z8=";
    };

    native-build-inputs = [ cmake ];

    cmake-flags = [
      "-DMDSPAN_ENABLE_TESTS=OFF"
      "-DMDSPAN_ENABLE_EXAMPLES=OFF"
      "-DMDSPAN_ENABLE_BENCHMARKS=OFF"
    ];

    post-install = ''
      install -m644 ${mdspan-shim} $out/include/mdspan
    '';

    meta = {
      description = "Reference implementation of P0009 std::mdspan";
      homepage = "https://github.com/kokkos/mdspan";
      license = [
        lib.licenses.asl20
        lib.licenses.bsd3
      ];
    };
  }
)
