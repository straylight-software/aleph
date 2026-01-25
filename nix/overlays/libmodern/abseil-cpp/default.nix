# nix/overlays/libmodern/abseil-cpp/default.nix
#
# Abseil C++ libraries - combined static archive
# https://abseil.io/
#
# Abseil produces ~130 separate static libraries. We combine them into
# a single libabseil.a with a clean pkg-config file.
#
# NOTE: mk-static-cpp already accepts lisp-case attrs and handles translation.
#
{
  final,
  lib,
  mk-static-cpp,
  combine-archive,
}:
mk-static-cpp {
  pname = "abseil-cpp";
  version = "20250127.1";

  src = final."fetchFromGitHub" {
    owner = "abseil";
    repo = "abseil-cpp";
    tag = "20250127.1";
    hash = "sha256-QTywqQCkyGFpdbtDBvUwz9bGXxbJs/qoFKF6zYAZUmQ=";
  };

  # Disable inline namespace versioning for cleaner symbols
  post-patch = ''
    sed -i 's/#define ABSL_OPTION_USE_INLINE_NAMESPACE 1/#define ABSL_OPTION_USE_INLINE_NAMESPACE 0/' absl/base/options.h
  '';

  build-inputs = [ final.gtest ];

  cmake-flags = [
    (lib.cmakeBool "ABSL_BUILD_TEST_HELPERS" true)
    (lib.cmakeBool "ABSL_USE_EXTERNAL_GOOGLETEST" true)
  ];

  # Combine all libabsl_*.a into single libabseil.a
  # Uses typed Haskell script instead of bash
  post-install = ''
    ${combine-archive}/bin/combine-archive $out ${final.stdenv.cc.bintools.targetPrefix}
  '';

  meta = {
    description = "Abseil C++ libraries (combined static archive)";
    homepage = "https://abseil.io/";
    license = lib.licenses.asl20;
  };
}
