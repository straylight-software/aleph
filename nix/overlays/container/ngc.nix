# nix/overlays/container/ngc.nix
#
# NGC (NVIDIA GPU Cloud) container Python extraction
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
  # Extract Python environment from NGC container
  #
  # Example:
  #   mk-ngc-python {
  #     name = "pytorch-python";
  #     container = mk-oci-rootfs {
  #       name = "pytorch-rootfs";
  #       ref = "nvcr.io/nvidia/pytorch:25.01-py3";
  #       hash = "sha256-...";
  #     };
  #     runtime-inputs = [ final.cudaPackages.cudatoolkit ];
  #   }
  #
  # Returns a derivation with:
  #   - $out/bin/python3 (patched interpreter)
  #   - $out/lib/python3.X/site-packages (from container)
  #   - Wrapper that sets PYTHONPATH and LD_LIBRARY_PATH
  #
  mk-ngc-python =
    {
      name,
      container,
      runtime-inputs ? [ ],
      python-version ? "3.12",
      extra-site-packages ? [ ],
    }:
    let
      run-path = aleph-lib.elf.mk-rpath (
        runtime-inputs
        ++ [
          final.stdenv.cc.cc.lib
          final.zlib
          final.bzip2
          final.xz
          final.libffi
          final.ncurses
          final.readline
          final.openssl
        ]
      );
    in
    stdenv.default {
      inherit name;
      src = container;

      native-build-inputs = [
        final.autoPatchelfHook
        final.makeWrapper
        final.patchelf
        final.file
      ];

      build-inputs = runtime-inputs ++ [
        final.stdenv.cc.cc.lib
        final.zlib
        final.bzip2
        final.xz
        final.libffi
        final.ncurses
        final.readline
        final.openssl
      ];

      dont-configure = true;
      dont-build = true;

      install-phase = ''
        runHook preInstall

        mkdir -p $out/{bin,lib}

        # Find and copy Python interpreter
        PYTHON_BIN=$(find $src -path "*/bin/python${python-version}" -type f | head -1)
        if [ -z "$PYTHON_BIN" ]; then
          PYTHON_BIN=$(find $src -path "*/bin/python3" -type f | head -1)
        fi

        if [ -n "$PYTHON_BIN" ]; then
          cp "$PYTHON_BIN" $out/bin/python3
          chmod +x $out/bin/python3
          ln -s python3 $out/bin/python
        fi

        # Copy Python standard library
        PYTHON_LIB=$(find $src -type d -name "python${python-version}" -path "*/lib/*" | head -1)
        if [ -n "$PYTHON_LIB" ]; then
          cp -a "$PYTHON_LIB" $out/lib/
        fi

        # Copy site-packages from all locations
        mkdir -p $out/lib/python${python-version}/site-packages
        for site in \
          $src/usr/lib/python3/dist-packages \
          $src/usr/local/lib/python${python-version}/dist-packages \
          $src/usr/lib/python${python-version}/site-packages
        do
          if [ -d "$site" ]; then
            cp -a "$site"/* $out/lib/python${python-version}/site-packages/ 2>/dev/null || true
          fi
        done

        # Copy container's native libs that Python modules depend on
        mkdir -p $out/lib/native
        for lib_dir in $src/usr/lib/x86_64-linux-gnu $src/usr/local/lib; do
          if [ -d "$lib_dir" ]; then
            find "$lib_dir" -name "*.so*" -type f 2>/dev/null | while read -r f; do
              cp -a "$f" $out/lib/native/ 2>/dev/null || true
            done
          fi
        done

        runHook postInstall
      '';

      pre-fixup = ''
        addAutoPatchelfSearchPath $out/lib/native
        addAutoPatchelfSearchPath $out/lib/python${python-version}/lib-dynload
      '';

      post-fixup = ''
        # Create wrapper with proper environment
        if [ -f $out/bin/python3 ]; then
          wrapProgram $out/bin/python3 \
            --prefix PYTHONPATH : "$out/lib/python${python-version}/site-packages${
              lib.concatMapStrings (p: ":${p}") extra-site-packages
            }" \
            --prefix LD_LIBRARY_PATH : "$out/lib/native:${run-path}"
        fi
      '';

      passthru = {
        inherit container python-version;
        site-packages = "$out/lib/python${python-version}/site-packages";
      };

      auto-patchelf-ignore-missing-deps = [
        "libcuda.so.1"
        "libnvidia-ml.so.1"
        "libcudart.so.*"
      ];

      meta = {
        description = "Python environment extracted from NGC container";
        platforms = [ "x86_64-linux" ];
      };
    };
}
