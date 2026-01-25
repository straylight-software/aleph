# nix/overlays/libmodern/libsodium.nix
#
# libsodium - A modern, easy-to-use crypto library
# https://doc.libsodium.org/
#
# Note: Uses autotools, not cmake, so we can't use mk-static-cpp
#
{ final, lib }:
final.stdenv.mkDerivation (final-attrs: {
  pname = "libsodium";
  version = "1.0.20";

  src = final.fetchurl {
    url = "https://download.libsodium.org/libsodium/releases/libsodium-${final-attrs.version}.tar.gz";
    hash = "sha256-67Ze9spDkzPCu0GgwZkFhyiNoH9sf9B8s6GMwY0wzhk=";
  };

  "nativeBuildInputs" = [
    final.autoreconfHook
    final.pkg-config
  ];

  "separateDebugInfo" = false;
  "enableParallelBuilding" = true;

  # Stack protector interferes with some libsodium internals
  "hardeningDisable" = [ "stackprotector" ];

  "configureFlags" = [
    "--disable-ssp"
    "--disable-shared"
    "--enable-static"
    "--with-pic"
  ];

  env = {
    "NIX_CFLAGS_COMPILE" = "-fPIC";
    "SOURCE_DATE_EPOCH" = "315532800"; # 1980-01-01 for reproducibility
  };

  # Verify we only built static libraries
  "postInstall" = ''
    if find "$out" -name "*.so*" -o -name "*.dylib" -o -name "*.dll" | grep -q .; then
      echo "ERROR: Shared libraries found despite static-only configuration" >&2
      exit 1
    fi
  '';

  "doCheck" = false;

  passthru.tests.pkg-config = final.testers.testMetaPkgConfig final-attrs."finalPackage";

  meta = {
    description = "Modern and easy-to-use crypto library";
    homepage = "https://doc.libsodium.org/";
    license = lib.licenses.isc;
    "pkgConfigModules" = [ "libsodium" ];
    platforms = lib.platforms.all;
  };
})
