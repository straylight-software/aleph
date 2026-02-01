# Wrap executables with proper environment after autoPatchelf
lib_paths="$out/lib"
[ -d $out/lib64 ] && lib_paths="$lib_paths:$out/lib64"
[ -d $out/tensorrt_llm/lib ] && lib_paths="$lib_paths:$out/tensorrt_llm/lib"
lib_paths="$lib_paths:@runtimeLibraryPath@"

for exe in $out/bin/*; do
	if [ -f "$exe" ] && [ -x "$exe" ]; then
		wrapProgram "$exe" \
			--prefix LD_LIBRARY_PATH : "$lib_paths" \
			--prefix PYTHONPATH : "$out/python" 2>/dev/null || true
	fi
done
