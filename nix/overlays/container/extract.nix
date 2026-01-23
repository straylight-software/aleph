# nix/overlays/container/extract.nix
#
# Binary extraction and patching utilities
#
{
  final,
  lib,
  straylight-lib,
}:
{
  # Extract and patch binary packages
  #
  # Example:
  #   extract {
  #     pname = "tensorrt";
  #     version = "10.8";
  #     src = ./TensorRT.tar.gz;
  #     runtime-inputs = [ cudaPackages.cudatoolkit ];
  #   }
  #
  extract =
    {
      pname,
      version,
      src,
      runtime-inputs ? [ ],
      install ? "cp -a . $out/",
      post-install ? "",
      meta ? { },
    }:
    let
      run-path = straylight-lib.elf.mk-rpath runtime-inputs;
      interpreter-path = "$(cat ${final.stdenv.cc}/nix-support/dynamic-linker)";
    in
    final.stdenv.mkDerivation {
      inherit
        pname
        version
        src
        meta
        ;

      nativeBuildInputs = [
        final.patchelf
        final.file
        final.gnutar
        final.gzip
        final.xz
        final.unzip
      ];

      dontConfigure = true;
      dontBuild = true;
      dontUnpack = true;

      installPhase = ''
        runHook preInstall
        mkdir -p $out
        ${install}
        ${post-install}
        runHook postInstall
      '';

      fixupPhase = ''
        runHook preFixup
        find $out -type f \( -executable -o -name "*.so*" \) 2>/dev/null | while read -r f; do
          [ -L "$f" ] && continue
          file "$f" | grep -q ELF || continue
          if file "$f" | grep -q "executable"; then
            patchelf --set-interpreter "${interpreter-path}" "$f" 2>/dev/null || true
          fi
          existing=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
          combined="${run-path}:$out/lib:$out/lib64''${existing:+:$existing}"
          patchelf --set-rpath "$combined" "$f" 2>/dev/null || true
        done
        runHook postFixup
      '';
    };

  # Create a stub library that provides symbol definitions
  #
  # Example:
  #   mk-stub "libcuda.so.1" [ "cuInit" "cuDeviceGet" "cuCtxCreate" ]
  #
  mk-stub =
    name: symbols:
    let
      stub-src = builtins.toFile "stub.c" (lib.concatMapStringsSep "\n" (s: "void ${s}() {}") symbols);
    in
    final.stdenv.mkDerivation {
      pname = "${name}-stub";
      version = "1.0";
      dontUnpack = true;

      buildPhase = ''
        $CC -shared -o ${name} ${stub-src}
      '';

      installPhase = ''
        mkdir -p $out/lib
        cp ${name} $out/lib/
      '';

      meta = {
        description = "Stub library providing symbol definitions for linking";
      };
    };
}
