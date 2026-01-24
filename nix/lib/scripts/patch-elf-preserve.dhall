-- nix/lib/scripts/patch-elf-preserve.dhall
--
-- Patch ELF files preserving existing rpath
-- Environment variables are injected by render.dhall-with-vars

let out : Text = env:OUT as Text
let interpreterPatch : Text = env:INTERPRETER_PATCH as Text
let rpath : Text = env:RPATH as Text

in ''
find ${out} -type f \( -executable -o -name "*.so*" \) 2>/dev/null | while read -r f; do
  [ -L "$f" ] && continue
  file "$f" | grep -q ELF || continue
  ${interpreterPatch}
  existing=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
  combined="${rpath}:${out}/lib:${out}/lib64${"$"}{existing:+:$existing}"
  patchelf --set-rpath "$combined" "$f" 2>/dev/null || true
done
''
