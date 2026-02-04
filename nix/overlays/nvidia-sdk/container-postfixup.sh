# Wrap executables with proper environment after autoPatchelf
libPaths="$out/lib"
[ -d $out/lib64 ] && libPaths="$libPaths:$out/lib64"
[ -d $out/tensorrt_llm/lib ] && libPaths="$libPaths:$out/tensorrt_llm/lib"
libPaths="$libPaths:@runtimeLibPath@"

for exe in $out/bin/*; do
  if [ -f "$exe" ] && [ -x "$exe" ]; then
    wrapProgram "$exe" \
      --prefix LD_LIBRARY_PATH : "$libPaths" \
      --prefix PYTHONPATH : "$out/python" 2>/dev/null || true
  fi
done
