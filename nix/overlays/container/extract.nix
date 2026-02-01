# nix/overlays/container/extract.nix
#
# Binary extraction and patching utilities
#
{
  final,
  lib,
  aleph-lib,
}:
let
  inherit (final.aleph) stdenv;
in
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
      run-path = aleph-lib.elf.mk-rpath runtime-inputs;
      interpreter-path = "$(cat ${final.stdenv.cc}/nix-support/dynamic-linker)";
    in
    stdenv.default (
      {
        inherit
          pname
          version
          src
          meta
          ;

        native-build-inputs = [
          final.patchelf
          final.file
          final.gnutar
          final.gzip
          final.xz
          final.unzip
        ];

        dont-configure = true;
        dont-build = true;
        dont-unpack = true;

        install-phase = ''
          runHook preInstall
          mkdir -p $out
          ${install}
          ${post-install}
          runHook postInstall
        '';
      }
      // {
        # fixupPhase is not in translate-attrs - passed through as-is
        "fixupPhase" =
          builtins.replaceStrings [ "@interpreterPath@" "@runPath@" ] [ interpreter-path run-path ]
            (builtins.readFile ./scripts/extract-fixup.sh);
      }
    );

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
    stdenv.default {
      pname = "${name}-stub";
      version = "1.0";
      dont-unpack = true;

      build-phase = ''
        $CC -shared -o ${name} ${stub-src}
      '';

      install-phase = ''
        mkdir -p $out/lib
        cp ${name} $out/lib/
      '';

      meta = {
        description = "Stub library providing symbol definitions for linking";
      };
    };
}
