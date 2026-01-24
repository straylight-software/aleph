-- nix/overlays/container/scripts/extract-fixup.dhall
--
-- Fixup script for extracted container binaries (patchelf interpreter/rpath)
-- Environment variables are injected by render.dhall-with-vars

let interpreterPath : Text = env:INTERPRETER_PATH as Text
let runPath : Text = env:RUN_PATH as Text

in ''
runHook preFixup
find $out -type f \( -executable -o -name "*.so*" \) 2>/dev/null | while read -r f; do
	[ -L "$f" ] && continue
	file "$f" | grep -q ELF || continue
	if file "$f" | grep -q "executable"; then
		patchelf --set-interpreter "${interpreterPath}" "$f" 2>/dev/null || true
	fi
	existing=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
	combined="${runPath}:$out/lib:$out/lib64${"$"}{existing:+:$existing}"
	patchelf --set-rpath "$combined" "$f" 2>/dev/null || true
done
runHook postFixup
''
