# nix/packages/mdspan.nix — Kokkos reference implementation of std::mdspan
#
# P0009 mdspan - multidimensional array view for C++23
# GCC 15 doesn't ship it yet, so we use the Kokkos reference impl.
# Patched to inject into std:: namespace (not std::experimental::)
#
{
  lib,
  pkgs,
  cmake,
}:
let
  inherit (pkgs.aleph) stdenv;

  # External API alias
  fetch-from-github = pkgs.fetchFromGitHub;

  mdspan-shim = ./mdspan-shim.hpp;
in
stdenv.default {
  pname = "mdspan";
  version = "0.6.0";

  src = fetch-from-github {
    owner = "kokkos";
    repo = "mdspan";
    # Can't use final-attrs with stdenv.default functor
    rev = "mdspan-0.6.0";
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
