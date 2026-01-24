-- nix/modules/flake/build/scripts/cxx-wrappers.dhall
--
-- Shell hook for copying C++ wrapper scripts
-- Environment variables are injected by render.dhall-with-vars

let scriptsDir : Text = env:SCRIPTS_DIR as Text

in ''
cp ${scriptsDir}/cxx-wrapper.bash bin/cxx
chmod +x bin/cxx

# compile_commands.json generator for clangd/clang-tidy
cp ${scriptsDir}/compdb.bash bin/compdb
chmod +x bin/compdb
''
