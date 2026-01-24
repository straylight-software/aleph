# nix/packages/mdspan.nix — Kokkos reference implementation of std::mdspan
#
# P0009 mdspan - multidimensional array view for C++23
# GCC 15 doesn't ship it yet, so we use the Kokkos reference impl.
# Patched to inject into std:: namespace (not std::experimental::)
#
{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
}:
let
  mdspan-shim = ./mdspan-shim.hpp;
in
stdenv.mkDerivation (finalAttrs: {
  pname = "mdspan";
  version = "0.6.0";

  src = fetchFromGitHub {
    owner = "kokkos";
    repo = "mdspan";
    rev = "mdspan-${finalAttrs.version}";
    hash = "sha256-bwE+NO/n9XsWOp3GjgLHz3s0JR0CzNDernfLHVqU9Z8=";
  };

  nativeBuildInputs = [ cmake ];

  cmakeFlags = [
    "-DMDSPAN_ENABLE_TESTS=OFF"
    "-DMDSPAN_ENABLE_EXAMPLES=OFF"
    "-DMDSPAN_ENABLE_BENCHMARKS=OFF"
  ];

  postInstall = ''
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
})
