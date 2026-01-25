# nix/overlays/libmodern/fmt.nix
#
# fmt - A modern formatting library for C++
# https://fmt.dev/
#
{
  final,
  lib,
  mk-static-cpp,
}:
let
  # Import prelude for translate-attrs
  translations = import ../../prelude/translations.nix { inherit lib; };
  inherit (translations) translate-attrs;
in
mk-static-cpp (translate-attrs {
  pname = "fmt";
  version = "11.2.0";

  src = final."fetchFromGitHub" {
    owner = "fmtlib";
    repo = "fmt";
    rev = "11.2.0";
    hash = "sha256-sAlU5L/olxQUYcv8euVYWTTB8TrVeQgXLHtXy8IMEnU=";
  };

  # Disable _BitInt which isn't supported on all platforms
  env.CXXFLAGS = "-DFMT_USE_BITINT=0";

  do-check = false;

  meta = {
    description = "Small, safe and fast formatting library";
    homepage = "https://fmt.dev/";
    license = lib.licenses.mit;
  };
})
