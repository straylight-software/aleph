# Add search paths for autoPatchelf
[ -d $out/lib ] && addAutoPatchelfSearchPath $out/lib
[ -d $out/lib64 ] && addAutoPatchelfSearchPath $out/lib64
[ -d $out/nvvm/lib64 ] && addAutoPatchelfSearchPath $out/nvvm/lib64
[ -d $out/tensorrt_llm/lib ] && addAutoPatchelfSearchPath $out/tensorrt_llm/lib

# Build library path from runtime inputs
libPaths="$out/lib"
[ -d $out/lib64 ] && libPaths="$libPaths:$out/lib64"
[ -d $out/nvvm/lib64 ] && libPaths="$libPaths:$out/nvvm/lib64"
[ -d $out/tensorrt_llm/lib ] && libPaths="$libPaths:$out/tensorrt_llm/lib"
libPaths="$libPaths:@runtimeLibPath@"

echo "Setting RPATH on ELF files before autoPatchelf..."

# Pre-patch all ELF files with correct RPATH before autoPatchelf runs
# This helps autoPatchelf find dependencies and prevents silent failures
find $out -type f 2>/dev/null | while read -r f; do
	if file "$f" 2>/dev/null | grep -q "ELF"; then
		# Set interpreter for executables
		if file "$f" | grep -q "ELF.*executable"; then
			patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$f" 2>/dev/null || true
		fi
		# Set RPATH for all ELF files
		patchelf --set-rpath "$libPaths" "$f" 2>/dev/null || true
	fi
done
