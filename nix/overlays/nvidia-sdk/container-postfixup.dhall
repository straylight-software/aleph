-- nix/overlays/nvidia-sdk/container-postfixup.dhall
--
-- Post-fixup script for NVIDIA SDK container binaries
-- Environment variables are injected by render.dhall-with-vars

let runtimeLibPath : Text = env:RUNTIME_LIB_PATH as Text

in ''
# Wrap executables with proper environment after autoPatchelf
libPaths="$out/lib"
[ -d $out/lib64 ] && libPaths="$libPaths:$out/lib64"
[ -d $out/tensorrt_llm/lib ] && libPaths="$libPaths:$out/tensorrt_llm/lib"
libPaths="$libPaths:${runtimeLibPath}"

for exe in $out/bin/*; do
	if [ -f "$exe" ] && [ -x "$exe" ]; then
		wrapProgram "$exe" \
			--prefix LD_LIBRARY_PATH : "$libPaths" \
			--prefix PYTHONPATH : "$out/python" 2>/dev/null || true
	fi
done
''
