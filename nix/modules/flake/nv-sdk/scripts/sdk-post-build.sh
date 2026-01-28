#!/usr/bin/env bash
# sdk-post-build.sh - NVIDIA SDK symlinkJoin postBuild script

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: lib64 -> lib symlink
# Many NVIDIA tools expect lib64, but Nix uses lib
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ! -e "$out/lib64" ]; then
  ln -s lib "$out/lib64"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 13 removed texture_fetch_functions.h (deprecated)
# but clang's wrapper still expects it - symlink to replacement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ! -e "$out/include/texture_fetch_functions.h" ] && [ -e "$out/include/texture_indirect_functions.h" ]; then
  ln -s texture_indirect_functions.h "$out/include/texture_fetch_functions.h"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NVIDIA USER-SPACE DRIVER + NVML (and friends)
# - Keep real driver libs available for runtime/debugging.
# - Keep link-time stubs under $out/stubs like it was designed.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRIVER_PKG="@driverPkg@"

if [ -n "$DRIVER_PKG" ]; then

  # Expose the full driver bundle under a stable prefix.
  if [ ! -e "$out/driver" ]; then
    ln -s "$DRIVER_PKG" "$out/driver"
  fi

  # Prefer putting driver runtime libs on the standard lib path.
  for libdir in "$DRIVER_PKG/lib" "$DRIVER_PKG/lib64"; do
    if [ -d "$libdir" ]; then
      for soname in \
        libcuda.so.1 libcuda.so \
        libnvidia-ml.so.1 libnvidia-ml.so \
        libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so \
        libnvidia-fatbinaryloader.so.1 libnvidia-fatbinaryloader.so \
        libnvidia-compiler.so.1 libnvidia-compiler.so; do
        if [ -e "$libdir/$soname" ] && [ ! -e "$out/lib64/$soname" ]; then
          ln -s "$libdir/$soname" "$out/lib64/$soname"
        fi
      done
    fi
  done
fi

# Link-time stubs (compile/link against these; runtime comes from driver).
mkdir -p "$out/stubs/lib"
if [ ! -e "$out/stubs/lib64" ]; then
  ln -s lib "$out/stubs/lib64"
fi

# Collect stubs from CUDA toolkit and (optionally) the driver package.
STUB_DIRS=(
  "@cudatoolkit@/lib/stubs"
  "@cudatoolkit@/lib64/stubs"
)

if [ -n "$DRIVER_PKG" ]; then
  STUB_DIRS+=("$DRIVER_PKG/lib/stubs" "$DRIVER_PKG/lib64/stubs")
fi

for stubdir in "${STUB_DIRS[@]}"; do
  if [ -d "$stubdir" ]; then
    # Symlink all stubs into $out/stubs/lib (flat).
    for f in "$stubdir"/*; do
      if [ -e "$f" ]; then
        ln -sf "$f" "$out/stubs/lib/$(basename "$f")"
      fi
    done
  fi
done

# ensure common link names exist in stubs dir...
if [ -e "$out/stubs/lib/libcuda.so.1" ] && [ ! -e "$out/stubs/lib/libcuda.so" ]; then
  ln -s libcuda.so.1 "$out/stubs/lib/libcuda.so"
fi

if [ -e "$out/stubs/lib/libnvidia-ml.so.1" ] && [ ! -e "$out/stubs/lib/libnvidia-ml.so" ]; then
  ln -s libnvidia-ml.so.1 "$out/stubs/lib/libnvidia-ml.so"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Create pkg-config files if they don't exist
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mkdir -p "$out/lib/pkgconfig"
sed "s|@PREFIX@|$out|g" "@cudaPc@" >"$out/lib/pkgconfig/cuda.pc"
sed "s|@PREFIX@|$out|g" "@cudnnPc@" >"$out/lib/pkgconfig/cudnn.pc"
sed "s|@PREFIX@|$out|g" "@tensorrtPc@" >"$out/lib/pkgconfig/tensorrt.pc"
sed "s|@PREFIX@|$out|g" "@ncclPc@" >"$out/lib/pkgconfig/nccl.pc"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# // version // manifest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cp "@sdkManifest@" "$out/NVIDIA_SDK_MANIFEST"
echo "" >>"$out/NVIDIA_SDK_MANIFEST"
echo "Contents:" >>"$out/NVIDIA_SDK_MANIFEST"
echo "$(find "$out" -name '*.so' -o -name '*.a' 2>/dev/null | wc -l) libraries" >>"$out/NVIDIA_SDK_MANIFEST"
echo "$(find "$out/include" -name '*.h' -o -name '*.hpp' -o -name '*.cuh' 2>/dev/null | wc -l) headers" >>"$out/NVIDIA_SDK_MANIFEST"

echo "NVIDIA SDK build complete. See $out/NVIDIA_SDK_MANIFEST for details."
