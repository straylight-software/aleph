-- nix/modules/flake/build/scripts/lean-wrappers.dhall
--
-- Shell hook for copying Lean wrapper scripts
-- Environment variables are injected by render.dhall-with-vars

let scriptsDir : Text = env:SCRIPTS_DIR as Text

in ''
cp ${scriptsDir}/lean-wrapper.bash bin/lean
chmod +x bin/lean

cp ${scriptsDir}/leanc-wrapper.bash bin/leanc
chmod +x bin/leanc
''
