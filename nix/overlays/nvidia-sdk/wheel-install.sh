runHook preUnpack
unzip $src -d unpacked
runHook postUnpack

runHook preInstall
mkdir -p $out

@copyLibs@

@copyIncludes@

# Create lib64 symlink for compatibility
[ -d $out/lib ] && [ ! -e $out/lib64 ] && ln -s lib $out/lib64 || true

# Make writable for patchelf
chmod -R u+w $out 2>/dev/null || true

# Patch RPATH for portability (before autoPatchelfHook runs)
find $out -name "*.so*" -type f | while read f; do
	patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64' "$f" 2>/dev/null || true
done

runHook postInstall
