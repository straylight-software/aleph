# nix/overlays/libmodern/fmt.nix
#
# fmt - A modern formatting library for C++
# https://fmt.dev/
#
# NOTE: mk-static-cpp already accepts lisp-case attrs and handles translation.
#
{
  final,
  lib,
  mk-static-cpp,
}:
mk-static-cpp {
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
}
